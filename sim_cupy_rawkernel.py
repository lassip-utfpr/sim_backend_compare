# =======================
# Importacao de pacotes de uso geral
# =======================
import argparse
from time import time
import numpy as np
from sim_support import *
from sim_support.simulator import Simulator

# ======================
# Importacao de pacotes especificos para a implementacao do simulador
# ======================
import cupy
import cupyx
import cupyx.jit


# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorCupyRawkernel(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)
        
        # Define o nome do simulador
        self._name = "CuPy-rawkernel"
        
        
    def implementation(self):
        # ---------------------------------
        # Definicao das funcoes de kernel
        # ---------------------------------
        # Pressao
        @cupyx.jit.rawkernel()
        def pressure_kernel(vx, vy, pressure, kappa,
                            mdvx_dx, mdvy_dy,
                            a_x_h, b_x_h, k_x_h,
                            a_y, b_y, k_y,
                            coefs, idx_fd, source_term, idx_source, it,
                            dt, one_dx, one_dy, p_2,
                            nx, ny, ord):
            x, y = cupyx.jit.grid(2)
            x_i32 = cupy.int32(x)
            y_i32 = cupy.int32(y)
            
            last = ord - 1
            offset = ord - 1
            i_dix = -idx_fd[last, 2]
            i_dfx = nx - idx_fd[last, 0]
            i_diy = -idx_fd[last, 3]
            i_dfy = ny - idx_fd[last, 1]

            # Pressure
            p_2[0] = 0.0
            if(x_i32 >= i_dix and x_i32 < i_dfx and y_i32 >= i_diy and y_i32 < i_dfy):
                vdvx_dx = 0.0
                vdvy_dy = 0.0
                for c in range(0, ord):
                    vdvx_dx += coefs[c] * (vx[x_i32 + idx_fd[c, 0], y] - vx[x_i32 + idx_fd[c, 2], y]) * one_dx
                    vdvy_dy += coefs[c] * (vy[x, y_i32 + idx_fd[c, 1]] - vy[x, y_i32 + idx_fd[c, 3]]) * one_dy

                mdvx_dx_new = b_x_h[x_i32 - offset] * mdvx_dx[x, y] + a_x_h[x_i32 - offset] * vdvx_dx
                mdvy_dy_new = b_y[y_i32 - offset] * mdvy_dy[x, y] + a_y[y_i32 - offset] * vdvy_dy

                vdvx_dx = vdvx_dx/k_x_h[x_i32 - offset] + mdvx_dx_new
                vdvy_dy = vdvy_dy/k_y[y_i32 - offset]  + mdvy_dy_new

                mdvx_dx[x, y] = mdvx_dx_new
                mdvy_dy[x, y] = mdvy_dy_new

                pressure[x, y] += kappa[x, y]*(vdvx_dx + vdvy_dy) * dt
                
                # Adiciona o sinal de fonte, se o pixel fizer parte de uma fonte
                idx_src = idx_source[x, y]
                if idx_src != -1:
                    pressure[x, y] += source_term[it - 1, idx_src] * dt * one_dx * one_dy
        
        # Velocidades
        @cupyx.jit.rawkernel()
        def velocity_kernel(vx, vy, pressure, rho_grid_vx, rho_grid_vy,
                            mdpressure_dx, mdpressure_dy,
                            a_x, b_x, k_x,
                            a_y_h, b_y_h, k_y_h,
                            coefs, idx_fd, sens_pressure, idx_sen, delay_rec, it,
                            dt, one_dx, one_dy, p_2,
                            nx, ny, ord):
            x, y = cupyx.jit.grid(2)
            x_i32 = cupy.int32(x)
            y_i32 = cupy.int32(y)
            
            last = ord - 1
            offset = ord - 1
            i_dix = -idx_fd[last, 3]
            i_dfx = nx - idx_fd[last, 1]
            i_diy = -idx_fd[last, 3]
            i_dfy = ny - idx_fd[last, 1]
            
            # Velocidade Vx
            if(x_i32 >= i_dix and x_i32 < i_dfx and y_i32 >= i_diy and y_i32 < i_dfy):
                dpressure_dx = 0.0

                for c in range(0, ord):
                    dpressure_dx += coefs[c] * (pressure[x_i32 + idx_fd[c, 1], y] - pressure[x_i32 + idx_fd[c, 3], y]) * one_dx

                mdpressure_dx_new = b_x[x_i32 - offset] * mdpressure_dx[x, y] + a_x[x_i32 - offset] * dpressure_dx
                dpressure_dx = dpressure_dx / k_x[x_i32 - offset] + mdpressure_dx_new
                mdpressure_dx[x, y] = mdpressure_dx_new

                vx[x, y] += dt * (dpressure_dx / rho_grid_vx[x, y])
            else:
                # Condicao de Dirichlet
                vx[x, y] = 0.0
            
            # Velocidade Vy
            i_dix = -idx_fd[last, 2]
            i_dfx = nx - idx_fd[last, 0]
            i_diy = -idx_fd[last, 2]
            i_dfy = ny - idx_fd[last, 0]
            
            if(x_i32 >= i_dix and x_i32 < i_dfx and y_i32 >= i_diy and y_i32 < i_dfy):
                dpressure_dy = 0.0

                for c in range(0, ord):
                    dpressure_dy += coefs[c] * (pressure[x, y_i32 + idx_fd[c, 0]] - pressure[x, y_i32 + idx_fd[c, 2]]) * one_dy

                mdpressure_dy_new = b_y_h[y_i32 - offset] * mdpressure_dy[x, y] + a_y_h[y_i32 - offset] * dpressure_dy
                dpressure_dy = dpressure_dy / k_y_h[y_i32 - offset] + mdpressure_dy_new
                mdpressure_dy[x, y] = mdpressure_dy_new

                vy[x, y] += dt * (dpressure_dy / rho_grid_vy[x, y])
            else:
                # Condicao de Dirichlet
                vy[x, y] = 0.0
                
            # Calcula a norma L2 da pressao
            p_2_old = p_2[0]
            p_2_new = cupy.abs(pressure[x, y])
            p_2[0] = p_2_old if p_2_old > p_2_new else p_2_new
            
            # Armazena o sinal do sensor, se o pixel fizer parte de um receptor
            sensor = idx_sen[x, y]
            if sensor != -1 and it >= delay_rec[sensor]:
                sens_pressure[it - 1, sensor] += pressure[x, y]

        # ---------------------------------
        # Implementacao do simulador
        # ---------------------------------
        super().implementation()
        
        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # Transfere arrays de parametros para a GPU
        a_x_gpu = cupy.asarray(self._a_x.flatten())
        b_x_gpu = cupy.asarray(self._b_x.flatten())
        k_x_gpu = cupy.asarray(self._k_x.flatten())
        a_x_half_gpu = cupy.asarray(self._a_x_half.flatten())
        b_x_half_gpu = cupy.asarray(self._b_x_half.flatten())
        k_x_half_gpu = cupy.asarray(self._k_x_half.flatten())
        a_y_gpu = cupy.asarray(self._a_y.flatten())
        b_y_gpu = cupy.asarray(self._b_y.flatten())
        k_y_gpu = cupy.asarray(self._k_y.flatten())
        a_y_half_gpu = cupy.asarray(self._a_y_half.flatten())
        b_y_half_gpu = cupy.asarray(self._b_y_half.flatten())
        k_y_half_gpu = cupy.asarray(self._k_y_half.flatten())
        
        rho_grid_vx_gpu = cupy.asarray(self._rho_grid_vx)
        rho_grid_vy_gpu = cupy.asarray(self._rho_grid_vy)
        coefs_gpu = cupy.asarray(self._coefs)
        
        # Arrays para as variaveis de memoria do calculo
        memory_dvx_dx_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        memory_dvy_dy_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        memory_dpressure_dx_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        memory_dpressure_dy_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))

        # Arrays dos campos de velocidade e pressoes
        vx_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        vy_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        pressure_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        pressure_l2_norm_gpu = cupy.asarray(np.zeros(1, dtype=flt32))
        
        # Arrays para os sensores
        sens_pressure_gpu = cupy.asarray(np.zeros((self._n_steps, self._n_rec), dtype=flt32))
        idx_sen_gpu = cupy.asarray(self._pos_sensors)
        delay_rec_gpu = cupy.asarray(self._delay_recv)

        # Calculo dos indices para o staggered grid
        ord = self._coefs.shape[0]
        idx_fd = np.array([[c + 1, c, -c, -c - 1] for c in range(ord)], dtype=int32)
        idx_fd_gpu = cupy.asarray(idx_fd, dtype=cupy.int32)

        # Definicao dos limites para a plotagem dos campos
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # Inicializa os mapas dos parametros de Lame
        kappa_gpu = cupy.asarray(self._rho_grid_vx * self._cp_grid_vx * self._cp_grid_vx)

        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            source_term = self._source_term[:, np.newaxis]
        source_term_gpu = cupy.asarray(source_term)
        idx_src_gpu = cupy.asarray(self._pos_sources)
        
        # Define os tamanhos dos blocos e dos grids para os kernels
        self._block_size_x = np.gcd(self._nx, 16)
        self._block_size_y = np.gcd(self._ny, 16)
        block_size = (self._block_size_x, self._block_size_y)
        grid_fields = ((self._nx + block_size[0] - 1) // block_size[0],
                       (self._ny + block_size[1] - 1) // block_size[1])
        
        # Laco de tempo para execucao da simulacao
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo da pressao
            pressure_kernel[grid_fields, block_size](vx_gpu, vy_gpu, pressure_gpu, kappa_gpu,
                                                     memory_dvx_dx_gpu, memory_dvy_dy_gpu,
                                                     a_x_half_gpu, b_x_half_gpu, k_x_half_gpu,
                                                     a_y_gpu, b_y_gpu, k_y_gpu,
                                                     coefs_gpu, idx_fd_gpu, source_term_gpu, idx_src_gpu, it,
                                                     self._dt, self._one_dx, self._one_dy, pressure_l2_norm_gpu,
                                                     self._nx, self._ny, ord)

            # Calculo das velocidades
            velocity_kernel[grid_fields, block_size](vx_gpu, vy_gpu, pressure_gpu, rho_grid_vx_gpu, rho_grid_vy_gpu,
                                                     memory_dpressure_dx_gpu, memory_dpressure_dy_gpu,
                                                     a_x_gpu, b_x_gpu, k_x_gpu,
                                                     a_y_half_gpu, b_y_half_gpu, k_y_half_gpu,
                                                     coefs_gpu, idx_fd_gpu, sens_pressure_gpu, idx_sen_gpu, delay_rec_gpu, it,
                                                     self._dt, self._one_dx, self._one_dy, pressure_l2_norm_gpu,
                                                     self._nx, self._ny, ord)

            psn2 = pressure_l2_norm_gpu.get()[0]
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f'Time step # {it} out of {self._n_steps}')
                    print(f'Max pressure = {psn2}')

                if self._show_anim:
                    self._windows_gpu[0].imv.setImage(pressure_gpu[ix_min:ix_max, iy_min:iy_max].get(),
                                                      levels=[self._min_val_fields, self._max_val_fields])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)
                
        sim_time = time() - t_gpu

        # Pega os resultados da simulacao
        pressure = pressure_gpu.get()
        sens_pressure = sens_pressure_gpu.get()
        
        # Libera a memoria alocada na GPU
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

        # --------------------------------------------
        # A funcao de implementacao do simulador deve retornar
        # um dicionario com as seguintes chaves:
        #   - "pressure": campo de pressao
        #   - "sens_pressure": sinais da pressao nos sensores
        #   - "gpu_str": string de identificacao da GPU utilizada na simulacao
        #   - "sim_time": tempo da simulacao, medido com a funcao time()
        #   - opcionalmente pode ter uma mensagem exclusiva da implementacao em "msg_impl"
        # --------------------------------------------
        return {"pressure": pressure, "sens_pressure": sens_pressure,
                "gpu_str": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), "sim_time": sim_time}


# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorCupyRawkernel(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")
    
except ValueError as value:
    print(value)
