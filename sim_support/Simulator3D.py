# =======================
# Importacao de pacotes de uso geral
# =======================
import numpy as np
import ast
import os.path
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
import pyqtgraph as pg
from sim_support import *
from sim_support.emission_law import EmissionLaw
from sim_support.windows_qt import Window
from sim_support.simul_classes import (SimulationROI, SimulationProbeLinearArray, SimulationProbePoint, SimulationProbeMonoCirc)
        

class Simulator3D:
    def __init__(self, file_config, sim_model="split", sim_type= "acoustic"):
        self._app = None
        self._device = None
        self._name = "simulator"
        self._sim_model = sim_model
        self._sim_type = sim_type

        # -----------------------
        # Leitura da configuracao no formato JSON
        # -----------------------
        with open(os.path.normpath(file_config), 'r') as f:
            self._configs = ast.literal_eval(f.read())
        
        # Configuracao da simulacao
        self._deriv_acc = self._configs.get("simul_params", 2).get("acc", 2)
        try:
            match self._sim_model:
                case "unsplit":
                    self._coefs = np.array(coefs_forward[self._deriv_acc - 2], dtype=flt32)
                case _:
                    self._coefs = np.array(coefs_Lui[self._deriv_acc - 2], dtype=flt32)
        except IndexError:
            print(f"Acurácia das derivadas {self._deriv_acc} não suportada. Usando o maior valor permitido (6).")
            match self._sim_model:
                case "unsplit":
                    self._coefs = np.array(coefs_forward[-1], dtype=flt32)
                case _:
                    self._coefs = np.array(coefs_Lui[-1], dtype=flt32)
        
        self._n_steps = self._configs.get("simul_params", 1000).get("time_steps", 1000)
        self._dt = flt32(self._configs.get("simul_params", 1.0).get("dt", 1.0))
        self._it_display = self._configs.get("simul_params", 10).get("it_display", 10)

        # Configuracao do corpo de prova
        self._cp = flt32(self._configs.get("specimen_params", 5.9).get("cp", 5.9))  # [mm/us]
        if "cp_map" in self._configs["specimen_params"]:
            self._cp_map = np.load(os.path.normpath(self._configs["specimen_params"]["cp_map"])).astype(flt32)
            
        self._cs = flt32(self._configs.get("specimen_params", 3.23).get("cs", 3.23))  # [mm/us]
        if "cs_map" in self._configs["specimen_params"]:
            self._cs_map = np.load(os.path.normpath(self._configs["specimen_params"]["cs_map"])).astype(flt32)

        self._rho = flt32(self._configs.get("specimen_params", 7800.0).get("rho", 7800.0))  # [mm/us]
        if "rho_map" in self._configs["specimen_params"]:
            self._rho_map = np.load(os.path.normpath(self._configs["specimen_params"]["rho_map"])).astype(flt32)

        # Configuracao da ROI
        pad = self._deriv_acc - 1
        if hasattr(self, "_rho_map"):
            self._roi = SimulationROI(**self._configs["roi"], pad=pad, rho_map=self._rho_map)
        else:
            self._roi = SimulationROI(**self._configs["roi"], pad=pad)
            
        # Escala do grid (valor do passo no espaco em milimetros)
        self._nx = self._roi.get_nx()
        self._ny = self._roi.get_ny()
        self._nz = self._roi.get_nz()
        self._dx = self._roi.get_dx()
        self._dy = self._roi.get_dy()
        self._dz = self._roi.get_dz()
        self._one_dx = flt32(1.0 / self._dx)
        self._one_dy = flt32(1.0 / self._dy)
        self._one_dz = flt32(1.0 / self._dz)
        
        # Inicializa os mapas de densidade do meio
        self._rho_grid_vx = np.ones((self._nx, self._ny, self._nz), dtype=flt32) * self._rho
        if hasattr(self, "_rho_map"):
            if self._rho_map.shape[0] < self._nx and self._rho_map.shape[1] < self._ny and self._rho_map.shape[2] < self._nz:
                self._rho_grid_vx[self._roi.get_ix_min(): self._roi.get_ix_max(),
                self._roi.get_iy_min(): self._roi.get_iy_max(),
                self._roi.get_iz_min(): self._roi.get_iz_max()] = self._rho_map
            elif self._rho_map.shape[0] > self._nx and self._rho_map.shape[1] > self._ny and self._rho_map.shape[2] > self._nz:
                self._rho_grid_vx = self._rho_map[:self._nx, :self._ny, :self._nz]
            elif self._rho_map.shape[0] == self._nx and self._rho_map.shape[1] == self._ny and self._rho_map.shape[2] == self._nz:
                self._rho_grid_vx = self._rho_map
            else:
                raise ValueError(f'rho_map shape {self._rho_map.shape} e incompativel com a ROI 3D')
            
        # rho_grid_vy - media de 8 pontos para o cubo 3D (staggered grid)
        self._rho_grid_vy = self._rho_grid_vx
        self._rho_grid_vy[:-1, :-1, :-1] = flt32(0.125) * (
            self._rho_grid_vx[:-1, :-1, :-1] + self._rho_grid_vx[1:, :-1, :-1] +
            self._rho_grid_vx[:-1, 1:, :-1] + self._rho_grid_vx[1:, 1:, :-1] +
            self._rho_grid_vx[:-1, :-1, 1:] + self._rho_grid_vx[1:, :-1, 1:] +
            self._rho_grid_vx[:-1, 1:, 1:] + self._rho_grid_vx[1:, 1:, 1:]
        )

         # cp_grid_vx e a matriz das velocidades longitudinais no mesmo grid de vx
        self._cp_grid_vx = np.ones((self._nx, self._ny, self._nz), dtype=flt32) * self._cp
        if hasattr(self, "_cp_map"):
            self._cp_grid_vx = np.ones((self._nx, self._ny), dtype=flt32) * self._cp
        if hasattr(self, "_cp_map"):
            if self._cp_map.shape[0] < self._nx and self._cp_map.shape[1] < self._ny and self._cp_map.shape[2] < self._nz:
                self._cp_grid_vx[self._roi.get_ix_min(): self._roi.get_ix_max(),
                self._roi.get_iy_min(): self._roi.get_iy_max(),                 
                self._roi.get_iz_min(): self._roi.get_iz_max()] = self._cp_map
            elif self._cp_map.shape[0] > self._nx and self._cp_map.shape[1] > self._ny and self._cp_map.shape[2] > self._nz:
                self._cp_grid_vx = self._cp_map[:self._nx, :self._ny, :self._nz]
            elif self._cp_map.shape[0] == self._nx and self._cp_map.shape[1] == self._ny and self._cp_map.shape[2] == self._nz:
                self._cp_grid_vx = self._cp_map
            else:
                raise ValueError(f'cp_map shape {self._cp_map.shape} e incompativel com a ROI')


        self._cs_grid_vx = np.ones((self._nx, self._ny, self._nz), dtype=flt32) * self._cs
        if hasattr(self, "_cs_map"):
            if self._cs_map.shape[0] < self._nx and self._cs_map.shape[1] < self._ny and self._cs_map.shape[2] < self._nz:
                self._cs_grid_vx[self._roi.get_ix_min(): self._roi.get_ix_max(),
                self._roi.get_iy_min(): self._roi.get_iy_max(),               
                self._roi.get_iz_min(): self._roi.get_iz_max()] = self._cs_map
            elif self._cs_map.shape[0] > self._nx and self._cs_map.shape[1] > self._ny and self._cs_map.shape[2] > self._nz:
                self._cs_grid_vx = self._cs_map[:self._nx, :self._ny, :self._nz]
            elif self._cs_map.shape[0] == self._nx and self._cs_map.shape[1] == self._ny and self._cs_map.shape[2] == self._nz:
                self._cs_grid_vx = self._cs_map
            else:
                raise ValueError(f'cs_map shape {self._cs_map.shape} e incompativel com a ROI')

        # Verifica a condicao de estabilidade de Courant
        # R. Courant et K. O. Friedrichs et H. Lewy (1928)
        cp_max = max(self._cp_grid_vx.max(), self._cp)
        courant_number = flt32(cp_max * self._dt * np.sqrt(self._one_dx ** 2 + self._one_dy ** 2 + self._one_dz ** 2))
        print(f'\nNumero de Courant e {courant_number}')
        if courant_number > 1.0:
            raise Exception("O passo de tempo e muito longo e a simulacao sera instavel", courant_number)

        # Configuracao dos transdutores
        self._probes = list()
        self._gain = flt32(0.0)
        probes_cfg = self._configs["probes"]
        for idx_p, p in enumerate(probes_cfg):
            if "linear" in p:
                self._probes.append(SimulationProbeLinearArray(**p["linear"], dec=self._roi.get_dec()))
            elif "point" in p:
                self._probes.append(SimulationProbePoint(**p["point"], dec=self._roi.get_dec()))
            elif "mono" in p:
                 self._probes.append(SimulationProbeMonoCirc(**p["mono"], dec=self._roi.get_dec()))

            self._gain = max(self._gain, flt32(self._probes[idx_p]._gain) if hasattr(self._probes[idx_p], "_gain") else 0.0)
                
        # Pega as listas de todos os pontos transmissores e receptores de todos os transdutores configurados
        i_probe_tx_ptos = list()
        i_probe_rx_ptos = list()
        delay_recv = list()
        self._n_rec = 0
        self._n_src = 0
        for pr in self._probes:
            i_probe_tx_ptos += pr.get_points_roi(self._roi, simul_type="3d", dir="e")[0]
            i_probe_rx_ptos += pr.get_points_roi(self._roi, simul_type="3d", dir="r")[0]
            delay_recv += pr.get_delay_rx()
            self._n_rec += pr.receivers.count(True)
            self._n_src += pr.emitters.count(True)

        # Posicao das fontes (Extração de X, Y e Z)
        i_probe_tx_ptos = np.array(i_probe_tx_ptos, dtype=int32).reshape(-1, 3)
        self._ix_src = i_probe_tx_ptos[:, 0].astype(int32)
        self._iy_src = i_probe_tx_ptos[:, 1].astype(int32)
        self._iz_src = i_probe_tx_ptos[:, 2].astype(int32)
        self._n_pto_src = i_probe_tx_ptos.shape[0]

        # Posicao dos receptores (Extração de X, Y e Z)
        i_probe_rx_ptos = np.array(i_probe_rx_ptos, dtype=int32).reshape(-1, 3)
        self._ix_rec = i_probe_rx_ptos[:, 0].astype(int32)
        self._iy_rec = i_probe_rx_ptos[:, 1].astype(int32)
        self._iz_rec = i_probe_rx_ptos[:, 2].astype(int32)
        
        # Calcula o delay de recepcao dos receptores
        self._delay_recv = (np.array(delay_recv) / self._dt + 1.0).astype(int32)
        
        # Calculo dos coeficientes de amortecimento para a PML, se PML estiver configurada
        # from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
        alpha_max_pml = flt32(2.0 * np.pi * (self._probes[0].get_freq() / 2.0))
        
        # Eixo X
        a_x, b_x, k_x = self._roi.calc_pml_array(axis='x', grid='f', dt=self._dt, cp=cp_max, alpha_max=alpha_max_pml)
        self._a_x = np.expand_dims(a_x.astype(flt32), axis=(1, 2))
        self._b_x = np.expand_dims(b_x.astype(flt32), axis=(1, 2))
        self._k_x = np.expand_dims(k_x.astype(flt32), axis=(1, 2))

        a_x_h, b_x_h, k_x_h = self._roi.calc_pml_array(axis='x', grid='h', dt=self._dt, cp=cp_max, alpha_max=alpha_max_pml)
        self._a_x_half = np.expand_dims(a_x_h.astype(flt32), axis=(1, 2))
        self._b_x_half = np.expand_dims(b_x_h.astype(flt32), axis=(1, 2))
        self._k_x_half = np.expand_dims(k_x_h.astype(flt32), axis=(1, 2))

        # Eixo Y (Profundidade)
        a_y, b_y, k_y = self._roi.calc_pml_array(axis='y', grid='f', dt=self._dt, cp=cp_max, alpha_max=alpha_max_pml)
        self._a_y = np.expand_dims(a_y.astype(flt32), axis=(0, 2))
        self._b_y = np.expand_dims(b_y.astype(flt32), axis=(0, 2))
        self._k_y = np.expand_dims(k_y.astype(flt32), axis=(0, 2))

        a_y_h, b_y_h, k_y_h = self._roi.calc_pml_array(axis='y', grid='h', dt=self._dt, cp=cp_max, alpha_max=alpha_max_pml)
        self._a_y_half = np.expand_dims(a_y_h.astype(flt32), axis=(0, 2))
        self._b_y_half = np.expand_dims(b_y_h.astype(flt32), axis=(0, 2))
        self._k_y_half = np.expand_dims(k_y_h.astype(flt32), axis=(0, 2))

        # Eixo Z (Altura)
        a_z, b_z, k_z = self._roi.calc_pml_array(axis='z', grid='f', dt=self._dt, cp=cp_max, alpha_max=alpha_max_pml)
        self._a_z = np.expand_dims(a_z.astype(flt32), axis=(0, 1))
        self._b_z = np.expand_dims(b_z.astype(flt32), axis=(0, 1))
        self._k_z = np.expand_dims(k_z.astype(flt32), axis=(0, 1))

        a_z_h, b_z_h, k_z_h = self._roi.calc_pml_array(axis='z', grid='h', dt=self._dt, cp=cp_max, alpha_max=alpha_max_pml)
        self._a_z_half = np.expand_dims(a_z_h.astype(flt32), axis=(0, 1))
        self._b_z_half = np.expand_dims(b_z_h.astype(flt32), axis=(0, 1))
        self._k_z_half = np.expand_dims(k_z_h.astype(flt32), axis=(0, 1))
        
        # Configuracoes Gerais
        self._n_iter = self._configs.get("simul_configs",1).get("n_iter", 1)
        self._max_val_fields = flt32(self._configs.get("simul_configs", 100.0).get("max_val_fields", 100.0))
        self._min_val_fields = flt32(self._configs.get("simul_configs", -100.0).get("min_val_fields", -100.0))
        self._show_anim = bool(self._configs.get("simul_configs", False).get("show_anim", False))
        self._show_debug = bool(self._configs.get("simul_configs", False).get("show_debug", False))
        self._show_figs = bool(self._configs.get("simul_configs", False).get("show_figs", False))
        self._plot_results = bool(self._configs.get("simul_configs", False).get("plot_results", False))
        self._plot_error = bool(self._configs.get("simul_configs", False).get("plot_error", False))
        self._plot_sensors = bool(self._configs.get("simul_configs", False).get("plot_sensors", False))
        self._plot_bscan = bool(self._configs.get("simul_configs", False).get("plot_bscan", False))
        self._save_results = bool(self._configs.get("simul_configs", False).get("save_results", False))
        self._save_field = bool(self._configs.get("simul_configs", False).get("save_field", False))
        self._save_sensors = bool(self._configs.get("simul_configs", False).get("save_sensors", False))
        self._save_bscan = bool(self._configs.get("simul_configs", False).get("save_bscan", False))
        self._save_sources = bool(self._configs.get("simul_configs", False).get("save_sources", False))
        self._source_env = bool(self._configs.get("simul_configs", False).get("source_env", False))
        self._bscan_xy = self._configs.get("simul_configs", False).get("bscan_xy", False)
        self._bscan_xz = self._configs.get("simul_configs", False).get("bscan_xz", False)
        self._bscan_yz = self._configs.get("simul_configs", False).get("bscan_yz", False)
        
        if ("emission_laws" in self._configs["simul_configs"] and
            os.path.isfile(os.path.normpath(self._configs["simul_configs"]["emission_laws"]))):
            self._emission_laws, _ = EmissionLaw.read_law(os.path.normpath(self._configs["simul_configs"]["emission_laws"]))
        if ("results_dir" in self._configs["simul_configs"] and
            os.path.isdir(os.path.normpath(self._configs["simul_configs"]["results_dir"]))):
            self._results_dir = os.path.normpath(self._configs["simul_configs"]["results_dir"])
            
        # Obtem fontes e receptores dos transdutores
        source_term = list()
        idx_src = list()
        idx_rec = list()
        self._idx_src_offset = 0
        self._idx_rec_offset = 0
        ord_source = 2 if self._sim_model == "unsplit" else 1

        for _pr in self._probes:
            if "linear" in p:
                if self._source_env:
                    st = _pr.get_source_term(samples=self._n_steps, dt=self._dt, out='e')
                    _, i_src = _pr.get_points_roi(sim_roi=self._roi, simul_type="3d")
                else:
                    st = _pr.get_source_term(samples=self._n_steps, dt=self._dt)
                    _, i_src = _pr.get_points_roi(sim_roi=self._roi, simul_type="3d")
            elif "mono" in p:
                if self._source_env:
                    st = _pr.get_source_term(samples=self._n_steps, dt=self._dt, out='e')
                    _, i_src = _pr.get_points_roi(sim_roi=self._roi, simul_type="3d")
                else:
                    st = _pr.get_source_term(samples=self._n_steps, dt=self._dt)
                    _, i_src = _pr.get_points_roi(sim_roi=self._roi, simul_type="3d")
            else:
                if self._source_env:
                    st = _pr.get_source_term(samples=self._n_steps, dt=self._dt, out='e', ord_der=ord_source)
                    _, i_src = _pr.get_points_roi(sim_roi=self._roi, simul_type="3d")
                else:
                    st = _pr.get_source_term(samples=self._n_steps, dt=self._dt, ord_der=ord_source)
                    _, i_src = _pr.get_points_roi(sim_roi=self._roi, simul_type="3d")
                
            if len(i_src) > 0:
                source_term.append(st)
                idx_src += [np.array(_s) + self._idx_src_offset for _s in i_src]
                self._idx_src_offset += len(i_src)

            i_rec = _pr.get_idx_rec(sim_roi=self._roi, simul_type="3d")
            if len(i_rec) > 0:
                idx_rec += [np.array(_r) + self._idx_rec_offset for _r in i_rec]
                self._idx_rec_offset += len(i_rec)
                
        # Source terms
        if len(source_term) > 1:
            self._source_term = np.concatenate(source_term, axis=1)
        else:
            self._source_term = source_term[0]

        # Posicionamento na Grade 3D
        self._pos_sources = -np.ones((self._nx, self._ny, self._nz), dtype=int32)
        self._pos_sources[self._ix_src, self._iy_src, self._iz_src] = np.array(idx_src).astype(int32).flatten()

        self._pos_sensors = -np.ones((self._nx, self._ny, self._nz), dtype=int32)
        self._pos_sensors[self._ix_rec, self._iy_rec, self._iz_rec] = np.array(idx_rec).astype(int32).flatten()
        
        self._idx_rec = np.array(idx_rec).astype(np.int32).flatten()
                
        # info_rec_pt agora carrega 4 colunas (x, y, z, id)
        self._info_rec_pt = np.column_stack((self._ix_rec, self._iy_rec, self._iz_rec, self._idx_rec)).astype(int32)
        
        numbers = list(self._idx_rec)
        offset_sensors = [numbers[0]]
        for i in range(1, len(numbers)):
            if numbers[i] != numbers[i - 1]:
                offset_sensors.append(int32(i))
        self._offset_sensors = np.array(offset_sensors, dtype=int32)
        self._n_pto_rec = int32(len(numbers))

        if self._show_anim:
            nx = self._roi.get_len_x()
            ny = self._roi.get_len_y()
            nz = self._roi.get_len_z()
            self._app = pg.QtWidgets.QApplication([])

            x_pos = 200 + np.arange(3) * (nx + 50)
            y_pos = 100 + np.arange(3) * (ny + 50)
            z_pos = 100 + np.arange(3) * (nz + 50)
            if (sim_type == "viscoelastic" or "elastic"):
                windows_gpu_data = [
                    {"title": "Stress ZZ [GPU]", "geometry": (x_pos[0], z_pos[0],
                                                       self._roi.get_nx(), self._roi.get_nz())},
                ]
            else:
                windows_gpu_data = [
                    {"title": "Pressure", "geometry": (x_pos[0], y_pos[0], z_pos[0],
                                                       self._roi.get_nx(), self._roi.get_ny(), self._roi.get_nz())},
                ]
            self._windows_gpu = [Window(title=data["title"], geometry=data["geometry"]) for data in windows_gpu_data]
        else:
            self._app = None
            self._windows_gpu = list()
        
    # Funcao de execucao da simulacao
    def run(self):
        now = datetime.now()
        sim_times = list()
        sim_total_times = list()
        mse_values = list()
        for n in range(self._n_iter):
            print(f'Iteracao {n}')
            n_laws = self._emission_laws.shape[0] if hasattr(self, "_emission_laws") else 1
            for law in range(n_laws):
                print(f'\tLaw {law} of {n_laws}')
                if hasattr(self, "_emission_laws"):
                    for p in self._probes:
                        p.set_t0(self._emission_laws[law])
                
                # Roda o simulador
                t_inic_simul = time()
                results_dict = self.implementation()
                sim_time_tot = time() - t_inic_simul
                sim_times.append(results_dict["sim_time"])
                sim_total_times.append(sim_time_tot)
                
                # Imprime algumas informacoes e cria o nome base dos arquivos de resultados
                print(results_dict["gpu_str"])
                print(f'{sim_times[-1]:.3}s')
                print(f'Tempo total (inclui transferencia de dados): {sim_total_times[-1]:.3}s')
                result_dir = self._results_dir if hasattr(self, "_results_dir") else os.path.join(".")
                name = os.path.join(result_dir, f'result_{self._name}_{now.strftime("%Y%m%d-%H%M%S")}_'
                        f'{self._nx}x{self._ny}_{self._n_steps}_iter_{n}_law_{law}')
                
                # Compara o resultado com a referencia (CPU-broadcast)
                try:
                    # Compara o resultado do campo de pressao com o valor de referência
                    pressure_ref = np.load(os.path.join(result_dir, f"result_ref_{self._sim_model}_field_pressure.npy"))[
                                               self._roi.get_ix_min():self._roi.get_ix_max(),
                                               self._roi.get_iz_min():self._roi.get_iz_max()]
                    
                    pressure = results_dict["pressure"][self._roi.get_ix_min():self._roi.get_ix_max(),
                            self._roi.get_iz_min():self._roi.get_iz_max()]
                    if pressure_ref.shape == pressure.shape:
                        error_pressure = pressure_ref - pressure
                        mse_pressure = np.mean(error_pressure ** 2)
                        if self._plot_error:
                            pressure_err_fig = plt.figure()
                            plt.title(f'{self._name} simulation pressure error - law ({law})\n({self._nx}x{self._ny})')
                            plt.imshow(error_pressure, aspect='auto', cmap='gray',
                                       extent=(self._roi.w_points[0], self._roi.w_points[-1],
                                               self._roi.h_points[-1], self._roi.h_points[0])
                                       )
                            plt.colorbar()
                    else:
                        mse_pressure = np.inf

                    # Compara o resultado dos sensores de pressao com o valores de referência
                    sens_pressure_ref = np.load(os.path.join(result_dir, f"result_ref_{self._sim_model}_bscan_pressure.npy"))
                    sens_pressure = results_dict["sens_pressure"]
                    if sens_pressure_ref.shape == sens_pressure.shape:
                        mse_sens_pressure = np.mean((sens_pressure_ref - sens_pressure) ** 2)
                    else:
                        mse_sens_pressure = np.inf
                        
                    print(f"MSE do campo de pressao em relacao a referencia: {mse_pressure:.4}")
                    print(f"MSE dos sensores de pressao em relacao a referencia: {mse_sens_pressure:.4}")
                    mse_values.append([mse_pressure, mse_sens_pressure])
                        
                except FileNotFoundError as err:
                    print(f"Arquivo {err} nao encontrado. Nao pode ser feita a comparacao com a referencia.")



                # Plota o mapa de pressao
                if self._sim_type == "acoustic":
                    bscan_ref = np.load(os.path.join('ensaios', 'ponto', 'results', 'result_ref_unsplit_bscan_pressure.npy'))
                    if self._plot_results:
                        pressure_sim_result = plt.figure()
                        plt.title(f'{self._name} simulation pressure - law ({law})\n({self._nx}x{self._ny})')
                        plt.imshow(results_dict["pressure"][self._roi.get_ix_min():self._roi.get_ix_max(), self._ny // 2,
                                self._roi.get_iz_min():self._roi.get_iz_max()].T,
                                aspect='auto', cmap='gray',
                                extent=(self._roi.w_points[0], self._roi.w_points[-1],
                                        self._roi.h_points[-1], self._roi.h_points[0]))
                        plt.colorbar()

                        if self._show_figs:
                            plt.show(block=False)

                        # Salva a imagem do campo de pressao
                        if self._save_results:
                            pressure_sim_result.savefig(name + '_field_pressure.png')

                    # Salva o campo de pressao
                    if self._save_results and self._save_field:
                        np.save(name + '_field_pressure', results_dict["pressure"])

                    # Plota individualmente os sinais tomados no sensores
                    if self._plot_sensors:
                        for r in range(results_dict["sens_pressure"].shape[1]):
                            sensor_pressure_result = plt.figure()
                            plt.title(f'{self._name} - Receptor {r + 1} - law ({law})')
                            plt.plot(results_dict["sens_pressure"][:, r], label='simulated')
                            plt.plot(bscan_ref, label='reference')
                            plt.legend()

                            # Pega a coordenada do primeiro emissor
                            for _pr in self._probes:
                                if all(val is True for val in _pr.emitters):
                                    coord_emitter = _pr.coord_center
                                    t0_emission = _pr._t0_emission
                                    break

                            # Pega a coordenada do primeiro receptor
                            for _pr in self._probes:
                                if all(val is True for val in _pr.receivers):
                                    coord_receiver = _pr.coord_center
                                    break

                            rd = np.sqrt(np.sum((coord_emitter - coord_receiver)**2))
                            td = rd / self._cp + t0_emission
                            ntd = td / self._dt
                            plt.plot([ntd, ntd], [np.min(results_dict["sens_pressure"][:, r]),
                                                  np.max(results_dict["sens_pressure"][:, r])],
                                     label="Posição esperada eco")

                            # Salva a imagem do sensor
                            if self._save_sensors:
                                    sensor_pressure_result.savefig(name + f'_sensor_{r}.png')

                        if self._show_figs:
                            plt.show(block=False)

                    if self._plot_bscan:
                        bscan_pressure_result = plt.figure()
                        plt.title(f'{self._name} simulation B-scan Pressure - law({law})\n({self._nx}x{self._ny}x{self._nz})')
                        plt.imshow(results_dict["sens_pressure"], aspect='auto', cmap='viridis')
                        plt.colorbar()

                        if self._show_figs:
                            plt.show(block=False)

                        # Salva a imagem b-scan dos valores dos sensores de pressao
                        if self._save_bscan:
                            bscan_pressure_result.savefig(name + '_bscan_pressure.png')

                    # Salva o array com os valores dos sensores de pressao
                    if self._save_bscan:
                        np.save(name + '_bscan_pressure', results_dict["sens_pressure"])
                else:
                    if self._plot_results:
                        stress_sim_result = plt.figure()
                        plt.title(f'{self._name} simulation stress - law ({law})\n({self._nx}x{self._ny})')
                        plt.imshow(results_dict["stress"][self._roi.get_ix_min():self._roi.get_ix_max(), self._ny // 2,
                                self._roi.get_iz_min():self._roi.get_iz_max()].T,
                                   aspect='auto', cmap='gray',
                                   extent=(self._roi.w_points[0], self._roi.w_points[-1],
                                           self._roi.h_points[-1], self._roi.h_points[0]))
                        plt.colorbar()

                        if self._show_figs:
                            plt.show(block=False)

                        # Salva a imagem do campo de pressao
                        if self._save_results:
                            stress_sim_result.savefig(name + '_field_stress.png')

                    # Salva o campo de estresse yy
                    if self._save_results and self._save_field:
                        np.save(name + '_field_stress_yy', results_dict["stress"])

                    # Plota individualmente os sinais tomados no sensores
                    if self._plot_sensors:
                        for r in range(results_dict["sens_stress"].shape[1]):
                            sensor_stress_result = plt.figure()
                            plt.title(f'{self._name} - Receptor {r + 1} - law ({law})')
                            plt.plot(results_dict["sens_stress"][:, r], label='simulated')
                            # plt.plot(bscan_ref, label='reference')
                            plt.legend()

                            # Pega a coordenada do primeiro emissor
                            for _pr in self._probes:
                                if all(val is True for val in _pr.emitters):
                                    coord_emitter = _pr.coord_center
                                    t0_emission = _pr._t0_emission
                                    break

                            # Pega a coordenada do primeiro receptor
                            for _pr in self._probes:
                                if all(val is True for val in _pr.receivers):
                                    coord_receiver = _pr.coord_center
                                    break

                            rd = np.sqrt(np.sum((coord_emitter - coord_receiver) ** 2))
                            td = rd / self._cp + t0_emission
                            ntd = td / self._dt
                            plt.plot([ntd, ntd], [np.min(results_dict["sens_stress"][:, r]),
                                                  np.max(results_dict["sens_stress"][:, r])],
                                     label="Posição esperada eco")

                            # Salva a imagem do sensor
                            if self._save_sensors:
                                sensor_pressure_result.savefig(name + f'_sensor_{r}.png')

                        if self._show_figs:
                            plt.show(block=False)

                    if self._plot_bscan:
                        bscan_pressure_result = plt.figure()
                        plt.title(f'{self._name} simulation B-scan Stress zz - law({law})\n({self._nx}x{self._ny}x{self._nz})')
                        plt.imshow(results_dict["sens_stress"], aspect='auto', cmap='viridis')
                        plt.colorbar()

                        if self._show_figs:
                            plt.show(block=False)

                        # Salva a imagem b-scan dos valores dos sensores de pressao
                        if self._save_bscan:
                            bscan_pressure_result.savefig(name + '_bscan_stress.png')

                    # Salva o array com os valores dos sensores de pressao
                    if self._save_bscan:
                        np.save(name + '_bscan_stress', results_dict["sens_stress"])


        sim_times = np.array(sim_times)
        sim_total_times = np.array(sim_total_times)
        mse_values = np.array(mse_values)

        print(f'TEMPO - {self._n_steps} pontos de tempo')
        if self._n_iter > 5:
            print(f'Tempo medio de execucao: {sim_times[5:].mean():.3}s (std = {sim_times[5:].std():.4})')
            print(f'Tempo medio total (inclui transferencia de dados): {sim_total_times[5:].mean():.3}s (std = {sim_total_times[5:].std():.4})')
            
        if self._n_iter > 5 and mse_values.shape[0] > 5:
            print(f'MSE medio do campo de pressao: {mse_values[5:, 0].mean():.4} (std = {mse_values[5:, 0].std():.4})')
            print(f'MSE medio dos sensores de pressao: {mse_values[5:, 1].mean():.4} (std = {mse_values[5:, 1].std():.4})')

        if self._save_sources:
            name = os.path.join(result_dir, f'sources_{self._name}_{now.strftime("%Y%m%d-%H%M%S")}')
            np.save(name, self._source_term)

        if self._save_results:
            name = os.path.join(result_dir, f'result_{self._name}_{now.strftime("%Y%m%d-%H%M%S")}_'
                        f'{self._nx}x{self._ny}_{self._n_steps}_iter_{n}_')
            np.savetxt(name + 'GPU_.csv', sim_times, '%10.3f', delimiter=',')

            with open(name + '_desc.txt', 'w') as f:
                f.write('Parametros do ensaio\n')
                f.write('--------------------\n')
                f.write('\n')
                f.write(f'Simulador: {self._name}\n')
                f.write(f'Modelo do simulador: {self._sim_model}\n')
                f.write(f'Quantidade de iteracoes no tempo: {self._n_steps}\n')
                f.write(f'Tamanho da ROI: {self._nx}x{self._ny}\n')
                f.write(f'GPU: {results_dict["gpu_str"]}\n')
                f.write(f'Numero de simulacoes: {self._n_iter}\n')
                if "msg_impl" in results_dict:
                    f.write(results_dict["msg_impl"])
                if self._n_iter > 5:
                    f.write(f'Tempo medio de execucao: {sim_times[5:].mean():.3}s\n')
                    f.write(f'Desvio padrao: {sim_times[5:].std():.4}\n')
                    f.write(f'Tempo medio total (inclui transferencia de dados): {sim_total_times[5:].mean():.3}s\n')
                    f.write(f'Desvio padrao: {sim_total_times[5:].std():.4}\n')
                    if mse_values.shape[0] > 5:
                        f.write(f'MSE medio do campo de pressao: {mse_values[5:, 0].mean():.4}\n')
                        f.write(f'Desvio padrao: {mse_values[5:, 0].std():.4}\n')
                        f.write(f'MSE medio dos sensores de pressao: {mse_values[5:, 1].mean():.4}\n')
                        f.write(f'Desvio padrao: {mse_values[5:, 1].std():.4}\n')
                else:
                    f.write(f'Tempo execucao: {sim_times[-1]:.3}s\n')
                    f.write(f'Tempo total (inclui transferencia de dados): {sim_total_times[-1]:.3}s\n')
                    if mse_values.shape[0] > 0:
                        f.write(f'MSE do campo de pressao: {mse_values[-1, 0]:.4}\n')
                        f.write(f'MSE medio dos sensores de pressao: {mse_values[-1, 1]:.4}\n')

        if self._show_figs:
            plt.show(block=False)
            
        if self._show_anim and self._app:
            self._app.exec()
   
    
    def implementation(self):
        print(f'Simulacao {self._name}')
        
        return dict()