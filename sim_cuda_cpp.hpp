#pragma once

#include <string>
#include <functional>
#include <vector>

namespace sim_cuda_cpp {

// Todos os vetores de grade usam layout row-major: indice = x * ny + y.
// Os vetores PML sao unidimensionais; o backend os indexa por x - (ord - 1)
// e y - (ord - 1), respectivamente. Informe os comprimentos para validacao.
struct SimulationInput {
    int nx = 0;
    int ny = 0;
    int n_steps = 0;
    int n_sources = 0;
    int n_receivers = 0;
    int ord = 0;  // Numero de coeficientes do stencil staggered.
    int it_display = 10;
    int pml_x_size = 0;
    int pml_y_size = 0;
    float dt = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;

    const float* coefficients = nullptr;       // ord elementos
    const float* rho_grid_vx = nullptr;        // nx * ny elementos
    const float* rho_grid_vy = nullptr;        // nx * ny elementos
    const float* cp_grid_vx = nullptr;         // nx * ny elementos

    const float* a_x = nullptr;                // pelo menos nx - 2*(ord-1) elementos
    const float* b_x = nullptr;
    const float* k_x = nullptr;
    const float* a_x_half = nullptr;
    const float* b_x_half = nullptr;
    const float* k_x_half = nullptr;
    const float* a_y = nullptr;                // pelo menos ny - 2*(ord-1) elementos
    const float* b_y = nullptr;
    const float* k_y = nullptr;
    const float* a_y_half = nullptr;
    const float* b_y_half = nullptr;
    const float* k_y_half = nullptr;

    const int* source_map = nullptr;           // nx * ny; -1 ou [0, n_sources)
    const float* source_term = nullptr;        // n_steps * n_sources, row-major
    const int* sensor_map = nullptr;           // nx * ny; -1 ou [0, n_receivers)
    const int* receiver_delay = nullptr;       // n_receivers elementos, em passos

    bool show_debug = false;
    std::function<void(const std::vector<float>&, int, float)> frame_callback;
};

struct SimulationOutput {
    std::vector<float> pressure;               // nx * ny elementos
    std::vector<float> sensor_pressure;        // n_steps * n_receivers, row-major
    float max_abs_pressure = 0.0f;
    std::string gpu_name;
    double sim_time_seconds = 0.0;
};

// Executa o backend split acústico 2D em CUDA. Os mapas, perfis PML,
// fontes e atrasos devem ser preparados pelo chamador antes da chamada.
SimulationOutput run(const SimulationInput& input);

}  // namespace sim_cuda_cpp
