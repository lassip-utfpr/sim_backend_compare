#include "sim_cuda_cpp.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr int nx = 8;
    constexpr int ny = 8;
    constexpr int n_steps = 3;
    constexpr int cells = nx * ny;

    std::vector<float> rho(cells, 1.0f);
    std::vector<float> cp(cells, 1.0f);
    std::vector<float> coefficient{1.0f, 0.0f};
    std::vector<float> zero_x(nx, 0.0f), one_x(nx, 1.0f);
    std::vector<float> zero_y(ny, 0.0f), one_y(ny, 1.0f);
    std::vector<int> sources(cells, -1), sensors(cells, -1);
    std::vector<float> source_term{1.0f, 0.0f, 0.0f};
    std::vector<int> receiver_delay{1};

    const int point = 3 * ny + 3;
    sources[point] = 0;
    sensors[point] = 0;

    sim_cuda_cpp::SimulationInput input;
    input.nx = nx;
    input.ny = ny;
    input.n_steps = n_steps;
    input.n_sources = 1;
    input.n_receivers = 1;
    input.ord = 2;
    input.pml_x_size = nx;
    input.pml_y_size = ny;
    input.dt = 1.0f;
    input.dx = 1.0f;
    input.dy = 1.0f;
    input.coefficients = coefficient.data();
    input.rho_grid_vx = rho.data();
    input.rho_grid_vy = rho.data();
    input.cp_grid_vx = cp.data();
    input.a_x = zero_x.data();
    input.b_x = one_x.data();
    input.k_x = one_x.data();
    input.a_x_half = zero_x.data();
    input.b_x_half = one_x.data();
    input.k_x_half = one_x.data();
    input.a_y = zero_y.data();
    input.b_y = one_y.data();
    input.k_y = one_y.data();
    input.a_y_half = zero_y.data();
    input.b_y_half = one_y.data();
    input.k_y_half = one_y.data();
    input.source_map = sources.data();
    input.source_term = source_term.data();
    input.sensor_map = sensors.data();
    input.receiver_delay = receiver_delay.data();

    try {
        const auto output = sim_cuda_cpp::run(input);
        if (output.pressure.size() != cells || output.sensor_pressure.size() != n_steps ||
            std::fabs(output.sensor_pressure[0] - 1.0f) > 1e-6f ||
            output.gpu_name.empty() || output.sim_time_seconds <= 0.0) {
            std::cerr << "Resultado inesperado no teste smoke CUDA\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "Falha no teste smoke CUDA: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
