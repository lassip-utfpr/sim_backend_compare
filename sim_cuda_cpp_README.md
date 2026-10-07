# Simulador acústico 2D completo em C++/CUDA

O executável `simulator_cuda_cpp` porta o fluxo de `Simulator` usado por `SimulatorCupyCuda`: carrega configuração, constrói ROI e mapas do material, configura sondas, fontes, sensores e CPML, roda o backend CUDA, compara referências e grava resultados. A API CUDA de baixo nível continua disponível em `sim_cuda_cpp.hpp` / `sim_cuda_cpp.cu`.

**Os kernels não são reimplementados no C++:** o CMake compila `sim_cupy_cuda.cu` diretamente como unidade CUDA da biblioteca e o wrapper chama os símbolos `pressure_kernel` e `velocity_kernel` definidos nesse arquivo. Não altere ou copie esses kernels para `sim_cuda_cpp.cu`; assim, ambos os backends consomem as mesmas definições-fonte. O wrapper usa a mesma regra de dimensão de bloco do CuPy (`gcd(nx, 16)` × `gcd(ny, 16)`), necessária porque o macro `xy` original não verifica o limite superior de `y`.

## Compilar

Dependências: CMake 3.24+, C++17, toolkit CUDA, JsonCpp, OpenCV (core, imgproc, imgcodecs e highgui) e GPU NVIDIA. Configure e compile pelo CMake Tools do VS Code. Os alvos de teste são `sim_cuda_cpp_smoke`, `simulator_cuda_cpp_tiny` e `simulator_cuda_cpp_linear`.

## Executar

Use `simulator_cuda_cpp --config <arquivo.json>` (ou `-c`). Sem argumento, procura `config.json`, como no Python. A implementação é o backend `split` de `SimulatorCupyCuda`; não implementa o modelo `unsplit` nem o backend CPU. Os caminhos de mapas, leis e resultados são resolvidos em relação ao diretório de trabalho atual, tal como no projeto Python.

## Funcionalidades portadas

- Parâmetros da simulação e stencil de diferenças finitas Lui; verificação Courant e monitoramento de estabilidade.
- ROI 2D x/z, padding, PML nos dois eixos, mapas `rho` e `cp` e perfis full/half-grid.
- Probes `point` e `linear`, flags TX/RX, atrasos, fontes gaussianas/derivadas e `source_env`.
- Leitura de arquivos `.law`, iterações e leis de emissão.
- Execução em CUDA, amostragem dos sensores, tempo de GPU e tempo total.
- Comparação de pressão/B-scan com referências `.npy` e cálculo de MSE.
- Salvamento de campo, fontes, B-scan, CSV de tempos e descrição do ensaio.
- Imagens de campo/erro/B-scan e sinais de sensores com OpenCV; animação da pressão e janelas opcionais.

Os `.npy` de mapas/referências podem ser `float32` ou `float64`, em C-order. Os arquivos de saída são gravados como `float32`. `show_figs` e `show_anim` precisam de uma sessão gráfica; para execução em servidor/headless, mantenha essas opções desativadas.

## Diferenças conhecidas

- Os gráficos são renderizados com OpenCV, não Matplotlib; a apresentação visual não é pixel a pixel idêntica.
- As leis de emissão são aplicadas ao regenerar os sinais para cada lei. O Python altera os atrasos das sondas depois de ter calculado `source_term`, então, no código atual, essa alteração não chega ao kernel.
- Canais TX/RX lineares são mapeados para índices sequenciais de canal; isso evita os índices inconsistentes em listas parciais ou elementos com mais de um ponto de grade.
- O alias de `rho_grid_vx`/`rho_grid_vy` do Python é mantido para compatibilidade numérica.
- `cs`/`cs_map` não alteram esta simulação acústica, assim como no backend Python split.
- A redução chamada de norma no kernel original tem corridas de escrita/leitura-modificação-escrita. Ela é mantida igual nos dois backends para não modificar o kernel solicitado; portanto, `max_abs_pressure`/a decisão de estabilidade pode não ser determinística, embora os campos da simulação sejam calculados pelos mesmos kernels.

Para executar apenas o núcleo CUDA diretamente, preencha `sim_cuda_cpp::SimulationInput` e chame `sim_cuda_cpp::run()`. Os mapas/perfis recebidos por essa API devem estar contíguos em row-major, com índice `x * ny + y`.
