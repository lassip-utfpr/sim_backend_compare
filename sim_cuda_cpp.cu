#include "sim_cuda_cpp.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

// As definicoes destas funcoes sao compiladas diretamente de sim_cupy_cuda.cu.
// Mantenha estas declaracoes sincronizadas com as assinaturas do arquivo CuPy.
extern "C" __global__ void pressure_kernel(
    float* vx, float* vy, float* pressure, float* kappa_unrelaxed,
    float* memory_dvx_dx, float* memory_dvy_dy, float* value_dvx_dx, float* value_dvy_dy,
    float* a_x_half, float* b_x_half, float* k_x_half,
    float* a_y, float* b_y, float* k_y,
    float* coefs, int* idx_fd, int* idx_src_gpu, float* source_term,
    int it, int nt, int n_pto_src, float dt, float one_dx, float one_dy,
    int nx, int ny, int ord, float* pressure_l2_norm_gpu);

extern "C" __global__ void velocity_kernel(
    float* vx, float* vy, float* pressure, float* rho_grid_vx, float* rho_grid_vy,
    float* memory_dpressure_dx, float* value_dpressure_dx,
    float* memory_dpressure_dy, float* value_dpressure_dy,
    float* a_x, float* b_x, float* k_x,
    float* a_y_half, float* b_y_half, float* k_y_half,
    float* coefs, int* idx_fd, float* sens_pressure, int* idx_sen_gpu, int* delay_rec,
    int it, int nt, int n_pto_rec, float dt, float one_dx, float one_dy,
    int nx, int ny, int ord, float* pressure_l2_norm_gpu);

namespace sim_cuda_cpp {
namespace {

constexpr float kStabilityThreshold = 1.0e38f;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ != 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)), "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

    void upload(const T* source, const char* name) {
        if (count_ == 0) return;
        if (source == nullptr) {
            throw std::invalid_argument(std::string("Ponteiro nulo: ") + name);
        }
        check_cuda(cudaMemcpy(data_, source, count_ * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }

    void download(T* destination, const char* name) const {
        if (count_ == 0) return;
        if (destination == nullptr) {
            throw std::invalid_argument(std::string("Ponteiro de destino nulo: ") + name);
        }
        check_cuda(cudaMemcpy(destination, data_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

void require_pointer(const void* pointer, const char* name) {
    if (pointer == nullptr) {
        throw std::invalid_argument(std::string("Ponteiro nulo: ") + name);
    }
}

void validate(const SimulationInput& in) {
    if (in.nx <= 0 || in.ny <= 0 || in.n_steps <= 0) {
        throw std::invalid_argument("nx, ny e n_steps devem ser positivos");
    }
    if (in.ord < 2 || in.ord > 6) {
        throw std::invalid_argument("ord deve estar entre 2 e 6");
    }
    if (in.n_sources < 0 || in.n_receivers < 0) {
        throw std::invalid_argument("n_sources e n_receivers nao podem ser negativos");
    }
    if (in.it_display <= 0) {
        throw std::invalid_argument("it_display deve ser positivo");
    }
    if (!std::isfinite(in.dt) || in.dt <= 0.0f || !std::isfinite(in.dx) || in.dx <= 0.0f ||
        !std::isfinite(in.dy) || in.dy <= 0.0f) {
        throw std::invalid_argument("dt, dx e dy devem ser finitos e positivos");
    }
    const int pml_x_required = in.nx - 2 * (in.ord - 1);
    const int pml_y_required = in.ny - 2 * (in.ord - 1);
    if (in.nx < 2 * in.ord || in.ny < 2 * in.ord ||
        in.pml_x_size < pml_x_required || in.pml_y_size < pml_y_required) {
        throw std::invalid_argument("Grade pequena ou vetores PML insuficientes para o stencil");
    }

    require_pointer(in.coefficients, "coefficients");
    require_pointer(in.rho_grid_vx, "rho_grid_vx");
    require_pointer(in.rho_grid_vy, "rho_grid_vy");
    require_pointer(in.cp_grid_vx, "cp_grid_vx");
    require_pointer(in.a_x, "a_x");
    require_pointer(in.b_x, "b_x");
    require_pointer(in.k_x, "k_x");
    require_pointer(in.a_x_half, "a_x_half");
    require_pointer(in.b_x_half, "b_x_half");
    require_pointer(in.k_x_half, "k_x_half");
    require_pointer(in.a_y, "a_y");
    require_pointer(in.b_y, "b_y");
    require_pointer(in.k_y, "k_y");
    require_pointer(in.a_y_half, "a_y_half");
    require_pointer(in.b_y_half, "b_y_half");
    require_pointer(in.k_y_half, "k_y_half");
    require_pointer(in.source_map, "source_map");
    require_pointer(in.sensor_map, "sensor_map");
    if (in.n_sources > 0) require_pointer(in.source_term, "source_term");
    if (in.n_receivers > 0) require_pointer(in.receiver_delay, "receiver_delay");
}

}  // namespace

SimulationOutput run(const SimulationInput& in) {
    validate(in);
    const std::size_t cells = static_cast<std::size_t>(in.nx) * in.ny;
    const std::size_t source_values = static_cast<std::size_t>(in.n_steps) * in.n_sources;
    const std::size_t sensor_values = static_cast<std::size_t>(in.n_steps) * in.n_receivers;
    const std::size_t pml_x = static_cast<std::size_t>(in.pml_x_size);
    const std::size_t pml_y = static_cast<std::size_t>(in.pml_y_size);

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    DeviceBuffer<float> d_coefficients(in.ord);
    DeviceBuffer<int> d_idx_fd(static_cast<std::size_t>(in.ord) * 4);
    DeviceBuffer<float> d_kappa(cells);
    DeviceBuffer<float> d_rho_vx(cells), d_rho_vy(cells), d_cp_vx(cells);
    DeviceBuffer<float> d_a_x(pml_x), d_b_x(pml_x), d_k_x(pml_x);
    DeviceBuffer<float> d_a_x_half(pml_x), d_b_x_half(pml_x), d_k_x_half(pml_x);
    DeviceBuffer<float> d_a_y(pml_y), d_b_y(pml_y), d_k_y(pml_y);
    DeviceBuffer<float> d_a_y_half(pml_y), d_b_y_half(pml_y), d_k_y_half(pml_y);
    DeviceBuffer<int> d_source_map(cells), d_sensor_map(cells), d_receiver_delay(in.n_receivers);
    DeviceBuffer<float> d_source_term(source_values), d_sensor_pressure(sensor_values);

    d_coefficients.upload(in.coefficients, "coefficients");
    std::vector<int> idx_fd(static_cast<std::size_t>(in.ord) * 4);
    for (int c = 0; c < in.ord; ++c) {
        idx_fd[static_cast<std::size_t>(c) * 4] = c + 1;
        idx_fd[static_cast<std::size_t>(c) * 4 + 1] = c;
        idx_fd[static_cast<std::size_t>(c) * 4 + 2] = -c;
        idx_fd[static_cast<std::size_t>(c) * 4 + 3] = -c - 1;
    }
    d_idx_fd.upload(idx_fd.data(), "idx_fd");
    d_rho_vx.upload(in.rho_grid_vx, "rho_grid_vx");
    d_rho_vy.upload(in.rho_grid_vy, "rho_grid_vy");
    d_cp_vx.upload(in.cp_grid_vx, "cp_grid_vx");
    std::vector<float> kappa(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        kappa[i] = in.rho_grid_vx[i] * in.cp_grid_vx[i] * in.cp_grid_vx[i];
    }
    d_kappa.upload(kappa.data(), "kappa_unrelaxed");
    d_a_x.upload(in.a_x, "a_x"); d_b_x.upload(in.b_x, "b_x"); d_k_x.upload(in.k_x, "k_x");
    d_a_x_half.upload(in.a_x_half, "a_x_half"); d_b_x_half.upload(in.b_x_half, "b_x_half");
    d_k_x_half.upload(in.k_x_half, "k_x_half");
    d_a_y.upload(in.a_y, "a_y"); d_b_y.upload(in.b_y, "b_y"); d_k_y.upload(in.k_y, "k_y");
    d_a_y_half.upload(in.a_y_half, "a_y_half"); d_b_y_half.upload(in.b_y_half, "b_y_half");
    d_k_y_half.upload(in.k_y_half, "k_y_half");
    d_source_map.upload(in.source_map, "source_map");
    d_sensor_map.upload(in.sensor_map, "sensor_map");
    d_source_term.upload(in.source_term, "source_term");
    d_receiver_delay.upload(in.receiver_delay, "receiver_delay");

    DeviceBuffer<float> d_vx(cells), d_vy(cells), d_pressure(cells);
    DeviceBuffer<float> d_memory_dvx_dx(cells), d_memory_dvy_dy(cells);
    DeviceBuffer<float> d_memory_dp_dx(cells), d_memory_dp_dy(cells);
    DeviceBuffer<float> d_value_dvx_dx(cells), d_value_dvy_dy(cells);
    DeviceBuffer<float> d_value_dp_dx(cells), d_value_dp_dy(cells);
    DeviceBuffer<float> d_pressure_l2_norm(1);
    check_cuda(cudaMemset(d_vx.get(), 0, cells * sizeof(float)), "zerar vx");
    check_cuda(cudaMemset(d_vy.get(), 0, cells * sizeof(float)), "zerar vy");
    check_cuda(cudaMemset(d_pressure.get(), 0, cells * sizeof(float)), "zerar pressure");
    check_cuda(cudaMemset(d_memory_dvx_dx.get(), 0, cells * sizeof(float)), "zerar memoria dvx/dx");
    check_cuda(cudaMemset(d_memory_dvy_dy.get(), 0, cells * sizeof(float)), "zerar memoria dvy/dy");
    check_cuda(cudaMemset(d_memory_dp_dx.get(), 0, cells * sizeof(float)), "zerar memoria dp/dx");
    check_cuda(cudaMemset(d_memory_dp_dy.get(), 0, cells * sizeof(float)), "zerar memoria dp/dy");
    check_cuda(cudaMemset(d_value_dvx_dx.get(), 0, cells * sizeof(float)), "zerar valor dvx/dx");
    check_cuda(cudaMemset(d_value_dvy_dy.get(), 0, cells * sizeof(float)), "zerar valor dvy/dy");
    check_cuda(cudaMemset(d_value_dp_dx.get(), 0, cells * sizeof(float)), "zerar valor dp/dx");
    check_cuda(cudaMemset(d_value_dp_dy.get(), 0, cells * sizeof(float)), "zerar valor dp/dy");
    if (sensor_values != 0) {
        check_cuda(cudaMemset(d_sensor_pressure.get(), 0, sensor_values * sizeof(float)), "zerar sensores");
    }

    const int block_x = std::gcd(in.nx, 16);
    const int block_y = std::gcd(in.ny, 16);
    const dim3 block(block_x, block_y);
    const dim3 grid(in.nx / block_x, in.ny / block_y);
    const float one_dx = 1.0f / in.dx;
    const float one_dy = 1.0f / in.dy;
    float max_abs_pressure = 0.0f;
    const auto simulation_start = std::chrono::steady_clock::now();

    for (int it = 1; it <= in.n_steps; ++it) {
        pressure_kernel<<<grid, block>>>(
            d_vx.get(), d_vy.get(), d_pressure.get(), d_kappa.get(),
            d_memory_dvx_dx.get(), d_memory_dvy_dy.get(), d_value_dvx_dx.get(), d_value_dvy_dy.get(),
            d_a_x_half.get(), d_b_x_half.get(), d_k_x_half.get(),
            d_a_y.get(), d_b_y.get(), d_k_y.get(), d_coefficients.get(), d_idx_fd.get(), d_source_map.get(),
            d_source_term.get(), it, in.n_steps, in.n_sources, in.dt, one_dx, one_dy,
            in.nx, in.ny, in.ord, d_pressure_l2_norm.get());
        check_cuda(cudaGetLastError(), "launch pressure_kernel");

        velocity_kernel<<<grid, block>>>(
            d_vx.get(), d_vy.get(), d_pressure.get(), d_rho_vx.get(), d_rho_vy.get(),
            d_memory_dp_dx.get(), d_value_dp_dx.get(), d_memory_dp_dy.get(), d_value_dp_dy.get(),
            d_a_x.get(), d_b_x.get(), d_k_x.get(), d_a_y_half.get(), d_b_y_half.get(), d_k_y_half.get(),
            d_coefficients.get(), d_idx_fd.get(), d_sensor_pressure.get(), d_sensor_map.get(), d_receiver_delay.get(),
            it, in.n_steps, in.n_receivers, in.dt, one_dx, one_dy, in.nx, in.ny, in.ord,
            d_pressure_l2_norm.get());
        check_cuda(cudaGetLastError(), "launch velocity_kernel");

        check_cuda(cudaMemcpy(&max_abs_pressure, d_pressure_l2_norm.get(), sizeof(float), cudaMemcpyDeviceToHost),
                   "ler max pressure");
        if (max_abs_pressure > kStabilityThreshold) {
            throw std::runtime_error("Simulacao tornando-se instavel; max |pressure| = " +
                                     std::to_string(max_abs_pressure));
        }
        if (in.show_debug && ((it % in.it_display) == 0 || it == 5)) {
            std::cout << "Time step " << it << " out of " << in.n_steps << '\n'
                      << "Max pressure = " << max_abs_pressure << '\n';
        }
        if (in.frame_callback && ((it % in.it_display) == 0 || it == 5)) {
            std::vector<float> frame(cells);
            d_pressure.download(frame.data(), "frame pressure");
            in.frame_callback(frame, it, max_abs_pressure);
        }
    }
    const auto simulation_end = std::chrono::steady_clock::now();

    SimulationOutput output;
    output.pressure.resize(cells);
    output.sensor_pressure.resize(sensor_values);
    d_pressure.download(output.pressure.data(), "pressure");
    d_sensor_pressure.download(output.sensor_pressure.data(), "sensor_pressure");
    output.max_abs_pressure = max_abs_pressure;
    output.gpu_name = properties.name;
    output.sim_time_seconds = std::chrono::duration<double>(simulation_end - simulation_start).count();
    return output;
}

}  // namespace sim_cuda_cpp
