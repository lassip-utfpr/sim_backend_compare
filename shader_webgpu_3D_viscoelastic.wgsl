// ++++++++++++++++++++++++++++++
// +++ Constants to override ++++
// ++++++++++++++++++++++++++++++
// Workgroup size for x dimension
const wsx : u32 = _WSX_;

// Workgroup size for y dimension
const wsy : u32 = _WSY_;

// Workgroup size for z dimension
const wsz : u32 = _WSZ_;

// Habilita vista do bscan xy
const bscan_xy : u32 = _BSCAN_XY_;

// Habilita vista do bscan xz
const bscan_xz : u32 = _BSCAN_XZ_;

// Habilita vista do bscan yz
const bscan_yz : u32 = _BSCAN_YZ_;

// Workgroup size for sensors store kernel
const idx_rec_offset: u32 = _IDX_REC_OFFSET_;


// +++++++++++++++++++++++++++++++
// +++ Index access functions ++++
// +++++++++++++++++++++++++++++++
// function to convert 2D [i,j] index into 1D [] index
fn ij(i: i32, j: i32, i_max: i32, j_max: i32) -> i32 {
    let index = j + i * j_max;

    return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max);
}

// function to convert 3D [i,j,k] index into 1D [] index
fn ijk(i: i32, j: i32, k: i32, i_max: i32, j_max: i32, k_max: i32) -> i32 {
    let index = k + j * k_max + i * k_max * j_max;

    return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max);
}

// function to convert 4D [i,j,k,l] index into 1D [] index
fn ijkl(i: i32, j: i32, k: i32, l: i32, i_max: i32, j_max: i32, k_max: i32, l_max: i32) -> i32 {
    let index = l 
              + k * l_max 
              + j * l_max * k_max 
              + i * l_max * k_max * j_max;

    let in_bounds = i >= 0 && i < i_max && 
                    j >= 0 && j < j_max && 
                    k >= 0 && k < k_max && 
                    l >= 0 && l < l_max;

    return select(-1, index, in_bounds);
}

// ++++++++++++++++++++++++++++++
// ++++ Group 0 - parameters ++++
// ++++++++++++++++++++++++++++++
struct SimIntValues {
    x_sz: i32,          // x field size
    y_sz: i32,          // y field size
    z_sz: i32,          // z field size
    n_iter: i32,        // num iterations
    n_src_el: i32,      // num probes tx elements
    n_rec_el: i32,      // num probes rx elements
    n_rec_pt: i32,      // num rec pto
    fd_coeff: i32,      // num fd coefficients
    it: i32,            // time iteraction
    n_sls: i32,         // num zener bodies
    visco_attn: i32     // viscoelastic attenuation flag
};

@group(0) @binding(0) // param_int32
var<storage,read_write> sim_int_par: SimIntValues;

// ----------------------------------

struct SimFltValues {
    dx: f32,            // delta x
    dy: f32,            // delta y
    dz: f32,            // delta z
    dt: f32,            // delta t
};

@group(0) @binding(1)   // param_flt32
var<storage,read> sim_flt_par: SimFltValues;

// -----------------------------------
// --- Force array access funtions ---
// -----------------------------------
@group(0) @binding(2) // source term
var<storage,read> source_term: array<f32>;

// function to get a source_term array value
fn get_source_term(n: i32, e: i32) -> f32 {
    let index: i32 = ij(n, e, sim_int_par.n_iter, sim_int_par.n_src_el);

    return select(0.0, source_term[index], index != -1);
}

// ----------------------------------

@group(0) @binding(3) // source term index
var<storage,read> idx_src: array<i32>;

// function to get a source term index of a source
fn get_idx_source_term(x: i32, y: i32, z: i32) -> i32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(-1, idx_src[index], index != -1);
}

// -------------------------------------------------
// --- CPML X coefficients array access funtions ---
// -------------------------------------------------
@group(0) @binding(4) // a_x
var<storage,read> a_x: array<f32>;

// function to get a a_x array value
fn get_a_x(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_x[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(5) // b_x
var<storage,read> b_x: array<f32>;

// function to get a b_x array value
fn get_b_x(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_x[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(6) // k_x
var<storage,read> k_x: array<f32>;

// function to get a k_x array value
fn get_k_x(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_x[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(7) // a_x_h
var<storage,read> a_x_h: array<f32>;

// function to get a a_x_h array value
fn get_a_x_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_x_h[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(8) // b_x_h
var<storage,read> b_x_h: array<f32>;

// function to get a b_x_h array value
fn get_b_x_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_x_h[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(9) // k_x_h
var<storage,read> k_x_h: array<f32>;

// function to get a k_x_h array value
fn get_k_x_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_x_h[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// -------------------------------------------------
// --- CPML Y coefficients array access funtions ---
// -------------------------------------------------
@group(0) @binding(10) // a_y
var<storage,read> a_y: array<f32>;

// function to get a a_y array value
fn get_a_y(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_y[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(11) // b_y
var<storage,read> b_y: array<f32>;

// function to get a b_y array value
fn get_b_y(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_y[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(12) // k_y
var<storage,read> k_y: array<f32>;

// function to get a k_y array value
fn get_k_y(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_y[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(13) // a_y_h
var<storage,read> a_y_h: array<f32>;

// function to get a a_y_h array value
fn get_a_y_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_y_h[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(14) // b_y_h
var<storage,read> b_y_h: array<f32>;

// function to get a b_y_h array value
fn get_b_y_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_y_h[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(15) // k_y_h
var<storage,read> k_y_h: array<f32>;

// function to get a k_y_h array value
fn get_k_y_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_y_h[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// -------------------------------------------------
// --- CPML Z coefficients array access funtions ---
// -------------------------------------------------
@group(0) @binding(16) // a_z
var<storage,read> a_z: array<f32>;

// function to get a a_z array value
fn get_a_z(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_z[n], n >= 0 && n < sim_int_par.z_sz - pad);
}

// ----------------------------------

@group(0) @binding(17) // b_z
var<storage,read> b_z: array<f32>;

// function to get a b_z array value
fn get_b_z(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_z[n], n >= 0 && n < sim_int_par.z_sz - pad);
}

// ----------------------------------

@group(0) @binding(18) // k_z
var<storage,read> k_z: array<f32>;

// function to get a k_z array value
fn get_k_z(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_z[n], n >= 0 && n < sim_int_par.z_sz - pad);
}

// ----------------------------------

@group(0) @binding(19) // a_z_h
var<storage,read> a_z_h: array<f32>;

// function to get a a_z_h array value
fn get_a_z_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_z_h[n], n >= 0 && n < sim_int_par.z_sz - pad);
}

// ----------------------------------

@group(0) @binding(20) // b_z_h
var<storage,read> b_z_h: array<f32>;

// function to get a b_z_h array value
fn get_b_z_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_z_h[n], n >= 0 && n < sim_int_par.z_sz - pad);
}

// ----------------------------------

@group(0) @binding(21) // k_z_h
var<storage,read> k_z_h: array<f32>;

// function to get a k_z_h array value
fn get_k_z_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_z_h[n], n >= 0 && n < sim_int_par.z_sz - pad);
}

// -------------------------------------------------------------
// --- Finite difference index limits arrays access funtions ---
// -------------------------------------------------------------
@group(0) @binding(22) // idx_fd
var<storage,read> idx_fd: array<i32>;

// function to get an index to ini-half grid
fn get_idx_ih(c: i32) -> i32 {
    let index: i32 = ij(c, 0, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get an index to ini-full grid
fn get_idx_if(c: i32) -> i32 {
    let index: i32 = ij(c, 1, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get an index to fin-half grid
fn get_idx_fh(c: i32) -> i32 {
    let index: i32 = ij(c, 2, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get an index to fin-full grid
fn get_idx_ff(c: i32) -> i32 {
    let index: i32 = ij(c, 3, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// ----------------------------------

@group(0) @binding(23) // fd_coeff
var<storage,read> fd_coeffs: array<f32>;

// function to get a fd coefficient
fn get_fdc(c: i32) -> f32 {
    return select(0.0, fd_coeffs[c], c >= 0 && c < sim_int_par.fd_coeff);
}

// ---------------------------------
// --- Rho map access funtions ---
// ---------------------------------
@group(0) @binding(24) // rho
var<storage,read> rho_map: array<f32>;

// function to get a rho value
fn get_rho(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, rho_map[index], index != -1);
}

// ---------------------------------
// --- Cp map access funtions ---
// ---------------------------------
@group(0) @binding(25) // cp
var<storage,read> cp_map: array<f32>;

// function to get a cp value
fn get_cp(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, cp_map[index], index != -1);
}

// ---------------------------------
// --- Cs map access funtions ---
// ---------------------------------
@group(0) @binding(26) // cs
var<storage,read> cs_map: array<f32>;

// function to get a cp value
fn get_cs(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, cs_map[index], index != -1);
}

@group(0) @binding(27) //somatorio alpha_p e alpha_s
var<storage,read> sum_alpha: array<f32>;

@group(0) @binding(28) //tau_epsilon_nu1,tau_sigma_nu1,tau_epsilon_nu2, tau_sigma_nu2 / o x indica qual tau e o y indica qual corpo zener
var<storage,read> tau_att: array<f32>;

fn get_tau(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, 4, sim_int_par.n_sls);

    return select(0.0, tau_att[index], index != -1);
}

// +++++++++++++++++++++++++++++++++++++
// ++++ Group 1 - simulation arrays ++++
// +++++++++++++++++++++++++++++++++++++
// ---------------------------------------
// --- Velocity arrays access funtions ---
// ---------------------------------------
@group(1) @binding(0) // vx field
var<storage,read_write> vx: array<f32>;

// function to get a vx array value
fn get_vx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, vx[index], index != -1);
}

// function to set a vx array value
fn set_vx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        vx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(1) // vy field
var<storage,read_write> vy: array<f32>;

// function to get a vy array value
fn get_vy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, vy[index], index != -1);
}

// function to set a vy array value
fn set_vy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        vy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(2) // vz field
var<storage,read_write> vz: array<f32>;

// function to get a vz array value
fn get_vz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, vz[index], index != -1);
}

// function to set a vz array value
fn set_vz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        vz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(3) // v_2
var<storage,read_write> v_2: f32;

// -------------------------------------
// --- Stress arrays access funtions ---
// -------------------------------------
@group(1) @binding(4) // sigmaxx field
var<storage,read_write> sigmaxx: array<f32>;

// function to get a sigmaxx array value
fn get_sigmaxx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, sigmaxx[index], index != -1);
}

// function to set a sigmaxx array value
fn set_sigmaxx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        sigmaxx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(5) // sigmayy field
var<storage,read_write> sigmayy: array<f32>;

// function to get a sigmayy array value
fn get_sigmayy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, sigmayy[index], index != -1);
}

// function to set a sigmayy array value
fn set_sigmayy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        sigmayy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(6) // sigmazz field
var<storage,read_write> sigmazz: array<f32>;

// function to get a sigmazz array value
fn get_sigmazz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, sigmazz[index], index != -1);
}

// function to set a sigmazz array value
fn set_sigmazz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        sigmazz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(7) // sigmaxy field
var<storage,read_write> sigmaxy: array<f32>;

// function to get a sigmaxy array value
fn get_sigmaxy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, sigmaxy[index], index != -1);
}

// function to set a sigmaxy array value
fn set_sigmaxy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        sigmaxy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(8) // sigmaxz field
var<storage,read_write> sigmaxz: array<f32>;

// function to get a sigmaxz array value
fn get_sigmaxz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, sigmaxz[index], index != -1);
}

// function to set a sigmaxz array value
fn set_sigmaxz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        sigmaxz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(9) // sigmayz field
var<storage,read_write> sigmayz: array<f32>;

// function to get a sigmayz array value
fn get_sigmayz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, sigmayz[index], index != -1);
}

// function to set a sigmayz array value
fn set_sigmayz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        sigmayz[index] = val;
    }
}

// -------------------------------------
// --- Memory arrays access funtions ---
// -------------------------------------
@group(1) @binding(10) // mdvx_dx field
var<storage,read_write> mdvx_dx: array<f32>;

// function to get a memory_dvx_dx array value
fn get_mdvx_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdvx_dx[index], index != -1);
}

// function to set a memory_dvx_dx array value
fn set_mdvx_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdvx_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(11) // mdvx_dy field
var<storage,read_write> mdvx_dy: array<f32>;

// function to get a memory_dvx_dy array value
fn get_mdvx_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdvx_dy[index], index != -1);
}

// function to set a memory_dvx_dy array value
fn set_mdvx_dy(x: i32, y: i32, z: i32, val : f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdvx_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(12) // mdvx_dz field
var<storage,read_write> mdvx_dz: array<f32>;

// function to get a memory_dvx_dz array value
fn get_mdvx_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdvx_dz[index], index != -1);
}

// function to set a memory_dvx_dz array value
fn set_mdvx_dz(x: i32, y: i32, z: i32, val : f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdvx_dz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(13) // mdvy_dx field
var<storage,read_write> mdvy_dx: array<f32>;

// function to get a memory_dvy_dx array value
fn get_mdvy_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdvy_dx[index], index != -1);
}

// function to set a memory_dvy_dx array value
fn set_mdvy_dx(x: i32, y: i32, z: i32, val : f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdvy_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(14) // mdvy_dy field
var<storage,read_write> mdvy_dy: array<f32>;

// function to get a memory_dvy_dy array value
fn get_mdvy_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdvy_dy[index], index != -1);
}

// function to set a memory_dvy_dy array value
fn set_mdvy_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdvy_dy[index] = val;
    }
}

@group(1) @binding(15) // mdvy_dz field
var<storage,read_write> mdvy_dz: array<f32>;

// function to get a memory_dvy_dz array value
fn get_mdvy_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdvy_dz[index], index != -1);
}

// function to set a memory_dvy_dz array value
fn set_mdvy_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdvy_dz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(16) // mdvz_dx field
var<storage,read_write> mdvz_dx: array<f32>;

// function to get a memory_dvz_dx array value
fn get_mdvz_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdvz_dx[index], index != -1);
}

// function to set a memory_dvz_dx array value
fn set_mdvz_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdvz_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(17) // mdvz_dy field
var<storage,read_write> mdvz_dy: array<f32>;

// function to get a memory_dvz_dy array value
fn get_mdvz_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdvz_dy[index], index != -1);
}

// function to set a memory_dvz_dy array value
fn set_mdvz_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdvz_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(18) // mdvz_dz field
var<storage,read_write> mdvz_dz: array<f32>;

// function to get a memory_dvz_dz array value
fn get_mdvz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdvz_dz[index], index != -1);
}

// function to set a memory_dvz_dz array value
fn set_mdvz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdvz_dz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(19) // mdsxx_dx field
var<storage,read_write> mdsxx_dx: array<f32>;

// function to get a memory_dsigmaxx_dx array value
fn get_mdsxx_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdsxx_dx[index], index != -1);
}

// function to set a memory_dsigmaxx_dx array value
fn set_mdsxx_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdsxx_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(20) // mdsyy_dy field
var<storage,read_write> mdsyy_dy: array<f32>;

// function to get a memory_dsigmayy_dy array value
fn get_mdsyy_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdsyy_dy[index], index != -1);
}

// function to set a memory_dsigmayy_dy array value
fn set_mdsyy_dy(x: i32, y: i32, z: i32, val : f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdsyy_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(21) // mdszz_dz field
var<storage,read_write> mdszz_dz: array<f32>;

// function to get a memory_dsigmazz_dz array value
fn get_mdszz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdszz_dz[index], index != -1);
}

// function to set a memory_dsigmazz_dz array value
fn set_mdszz_dz(x: i32, y: i32, z: i32, val : f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdszz_dz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(22) // mdsxy_dx field
var<storage,read_write> mdsxy_dx: array<f32>;

// function to get a memory_dsigmaxy_dx array value
fn get_mdsxy_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdsxy_dx[index], index != -1);
}

// function to set a memory_dsigmaxy_dx array value
fn set_mdsxy_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdsxy_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(23) // mdsxy_dy field
var<storage,read_write> mdsxy_dy: array<f32>;

// function to get a memory_dsigmaxy_dy array value
fn get_mdsxy_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdsxy_dy[index], index != -1);
}

// function to set a memory_dsigmaxy_dy array value
fn set_mdsxy_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdsxy_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(24) // mdsxz_dx field
var<storage,read_write> mdsxz_dx: array<f32>;

// function to get a memory_dsigmaxz_dx array value
fn get_mdsxz_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdsxz_dx[index], index != -1);
}

// function to set a memory_dsigmaxz_dx array value
fn set_mdsxz_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdsxz_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(25) // mdsxz_dz field
var<storage,read_write> mdsxz_dz: array<f32>;

// function to get a memory_dsigmaxz_dz array value
fn get_mdsxz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdsxz_dz[index], index != -1);
}

// function to set a memory_dsigmaxz_dz array value
fn set_mdsxz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdsxz_dz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(26) // mdsyz_dy field
var<storage,read_write> mdsyz_dy: array<f32>;

// function to get a memory_dsigmayz_dy array value
fn get_mdsyz_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdsyz_dy[index], index != -1);
}

// function to set a memory_dsigmayz_dy array value
fn set_mdsyz_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdsyz_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(27) // mdsyz_dz field
var<storage,read_write> mdsyz_dz: array<f32>;

// function to get a memory_dsigmayz_dz array value
fn get_mdsyz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    return select(0.0, mdsyz_dz[index], index != -1);
}

// function to set a memory_dsigmayz_dz array value
fn set_mdsyz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz);

    if(index != -1) {
        mdsyz_dz[index] = val;
    }
}

// ----------------------------------


@group(1) @binding(28) //r_xx
var<storage,read_write> r_xx: array<f32>;

fn get_r_xx(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);
                     

    return select(0.0, r_xx[index], index != -1);
}

// function to set a r_xx array value
fn set_r_xx(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(29) //r_yy
var<storage,read_write> r_yy: array<f32>;

fn get_r_yy(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_yy[index], index != -1);
}

// function to set a r_yy array value
fn set_r_yy(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_yy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(30) //r_zz
var<storage,read_write> r_zz: array<f32>;

fn get_r_zz(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_zz[index], index != -1);
}

// function to set a r_zz array value
fn set_r_zz(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_zz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(31) //r_xy
var<storage,read_write> r_xy: array<f32>;

fn get_r_xy(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_xy[index], index != -1);
}

// function to set a r_xy array value
fn set_r_xy(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(32) //r_xz
var<storage,read_write> r_xz: array<f32>;

fn get_r_xz(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_xz[index], index != -1);
}

// function to set a r_xz array value
fn set_r_xz(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(33) //r_yz
var<storage,read_write> r_yz: array<f32>;

fn get_r_yz(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_yz[index], index != -1);
}

// function to set a r_yz array value
fn set_r_yz(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_yz[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(34) //r_xx_old
var<storage,read_write> r_xx_old: array<f32>;

fn get_r_xx_old(x: i32, y: i32, l: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_xx_old[index], index != -1);
}

fn set_r_xx_old(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xx_old[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(35) //r_yy_old
var<storage,read_write> r_yy_old: array<f32>;

fn get_r_yy_old(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_yy_old[index], index != -1);
}

fn set_r_yy_old(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_yy_old[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(36) //r_zz_old
var<storage,read_write> r_zz_old: array<f32>;

fn get_r_zz_old(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_zz_old[index], index != -1);
}

fn set_r_zz_old(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_zz_old[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(37) //r_xy_old
var<storage,read_write> r_xy_old: array<f32>;

fn get_r_xy_old(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_xy_old[index], index != -1);
}

fn set_r_xy_old(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xy_old[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(38) //r_xz_old
var<storage,read_write> r_xz_old: array<f32>;

fn get_r_xz_old(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_xz_old[index], index != -1);
}

fn set_r_xz_old(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xz_old[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(39) //r_yz_old
var<storage,read_write> r_yz_old: array<f32>;

fn get_r_yz_old(x: i32, y: i32, z: i32, l: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    return select(0.0, r_yz_old[index], index != -1);
}

fn set_r_yz_old(x: i32, y: i32, z: i32, l: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, l, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_yz_old[index] = val;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Group 2 - sensors arrays and energies ++++
// +++++++++++++++++++++++++++++++++++++++++++++++
// --------------------------------------
// --- Sensors arrays access funtions ---
// --------------------------------------
@group(2) @binding(0) // sensors signals sigzz
var<storage,read_write> sensors_sigzz: array<f32>;

// function to get a sens_sigyy array value
fn get_sens_sigzz(n: i32, s: i32) -> f32 {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    return select(0.0, sensors_sigzz[index], index != -1);
}

// function to set a sens_sigyy array value
fn set_sens_sigzz(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    if(index != -1) {
        sensors_sigzz[index] = val;
    }
}

// ----------------------------------

@group(2) @binding(1) // delay sensor
var<storage,read> delay_rec: array<i32>;

// function to get a delay receiver value
fn get_delay_rec(s: i32) -> i32 {
    return select(0, delay_rec[s], s >= 0 && s < sim_int_par.n_rec_el);
}

// ----------------------------------

@group(2) @binding(2) // info rec ptos
var<storage,read> info_rec_pt: array<i32>;

// function to get a x-index of a receiver point
fn get_idx_x_sensor(n: i32) -> i32 {
    let index: i32 = ij(n, 0, sim_int_par.n_rec_pt, 4);

    return select(-1, info_rec_pt[index], index != -1);
}

// function to get a y-index of a receiver point
fn get_idx_y_sensor(n: i32) -> i32 {
    let index: i32 = ij(n, 1, sim_int_par.n_rec_pt, 4);

    return select(-1, info_rec_pt[index], index != -1);
}

// function to get a z-index of a receiver point
fn get_idx_z_sensor(n: i32) -> i32 {
    let index: i32 = ij(n, 2, sim_int_par.n_rec_pt, 4);

    return select(-1, info_rec_pt[index], index != -1);
}

// function to get a sensor-index of a receiver point
fn get_idx_sensor(n: i32) -> i32 {
    let index: i32 = ij(n, 3, sim_int_par.n_rec_pt, 4);

    return select(-1, info_rec_pt[index], index != -1);
}

// ----------------------------------

@group(2) @binding(3) // info rec ptos
var<storage,read> offset_sensors: array<i32>;

// function to get the offset of a sensor receiver in info_rec_pt table
fn get_offset_sensor(s: i32) -> i32 {
    return select(-1, offset_sensors[s], s >= 0 && s < sim_int_par.n_rec_el);
}

// ---------------
// --- Kernels ---
// ---------------
@compute
@workgroup_size(wsx, wsy, wsz)
fn teste_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // z thread index
    let dx: f32 = sim_flt_par.dx;
    let dy: f32 = sim_flt_par.dy;
    let dt: f32 = sim_flt_par.dt;
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset_x: i32 = sim_int_par.fd_coeff - 1;
    let offset_y: i32 = sim_int_par.fd_coeff - 1;


    // Normal stresses
    var id_x_i: i32 = -get_idx_fh(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        set_vx(x, y, z, get_rho(x, y, z));
        set_vy(x, y, z, get_cp(x, y, z));
        set_vz(x, y, z, get_cp(x, y, z));
        
    }
}

// Kernel to calculate stresses [sigmaxx, sigmayy, sigmaxy]
@compute
@workgroup_size(wsx, wsy, wsz)
fn sigma_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // z thread index
    let dx: f32 = sim_flt_par.dx;
    let dy: f32 = sim_flt_par.dy;
    let dz: f32 = sim_flt_par.dz;
    let dt: f32 = sim_flt_par.dt;
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset: i32 = sim_int_par.fd_coeff - 1;
    let visco_attn: bool = bool(sim_int_par.visco_attn);

    // Normal stresses
    var id_x_i: i32 = -get_idx_fh(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    var id_z_i: i32 = -get_idx_ff(last);
    var id_z_f: i32 = sim_int_par.z_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdvx_dx: f32 = 0.0;
        var vdvy_dy: f32 = 0.0;
        var vdvz_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvx_dx += get_fdc(c) * (get_vx(x + get_idx_ih(c), y, z) - get_vx(x + get_idx_fh(c), y, z)) / dx;
            vdvy_dy += get_fdc(c) * (get_vy(x, y + get_idx_if(c), z) - get_vy(x, y + get_idx_ff(c), z)) / dy;
            vdvz_dz += get_fdc(c) * (get_vz(x, y, z + get_idx_if(c)) - get_vz(x, y, z + get_idx_ff(c))) / dz;
        }

        var mdvx_dx_new: f32 = get_b_x_h(x - offset) * get_mdvx_dx(x, y, z) + get_a_x_h(x - offset) * vdvx_dx;
        var mdvy_dy_new: f32 = get_b_y(y - offset) * get_mdvy_dy(x, y, z) + get_a_y(y - offset) * vdvy_dy;
        var mdvz_dz_new: f32 = get_b_z(z - offset) * get_mdvz_dz(x, y, z) + get_a_z(z - offset) * vdvz_dz;

        vdvx_dx = vdvx_dx/get_k_x_h(x - offset) + mdvx_dx_new;
        vdvy_dy = vdvy_dy/get_k_y(y - offset)  + mdvy_dy_new;
        vdvz_dz = vdvz_dz/get_k_z(z - offset)  + mdvz_dz_new;

        set_mdvx_dx(x, y, z, mdvx_dx_new);
        set_mdvy_dy(x, y, z, mdvy_dy_new);
        set_mdvz_dz(x, y, z, mdvz_dz_new);

        let rho = get_rho(x, y, z);
        let cp = get_cp(x, y, z);
        let cs = get_cs(x, y, z);
        let lambda: f32 = rho * (cp * cp - 2.0 * cs * cs);
        let mu: f32 = rho * (cs * cs);
        let lambdaplus2mu: f32 = lambda + 2.0 * mu;
        let lambdaplusmu: f32 = lambda + mu;
        var sigmaxx: f32 = 0.0;
        var sigmayy: f32 = 0.0;
        var sigmazz: f32 = 0.0;

        if(visco_attn) {

            var sum_r_xx: f32 = 0.0;
            var sum_r_yy: f32 = 0.0;
            var sum_r_zz: f32 = 0.0;

            var div: f32 = vdvx_dx + vdvy_dy + vdvz_dz;

            for(var _l: i32 = 0; _l < sim_int_par.n_sls; _l++) {

                var inv_tau_sigma_p: f32 = 1.0/get_tau(1,_l);
                var inv_tau_sigma_s: f32 = 1.0/get_tau(3,_l);

                var alpha_p:f32 = get_tau(0,_l)/get_tau(1,_l);
                var alpha_s:f32 = get_tau(2,_l)/get_tau(3,_l);

                var deltat_phi_p: f32 = dt * (1.0 - alpha_p) * inv_tau_sigma_p / sum_alpha[0];
                var deltat_phi_s: f32 = dt * (1.0 - alpha_s) * inv_tau_sigma_s / sum_alpha[1];

                var half_deltat_overtau_sigma_p: f32 = 0.5 * dt * inv_tau_sigma_p;
                var half_deltat_overtau_sigma_s: f32 = 0.5 * dt * inv_tau_sigma_s;

                var mult_factor_tau_sigma_p: f32 = 1.0 / (1.0 + half_deltat_overtau_sigma_p);
                var mult_factor_tau_sigma_s: f32 = 1.0 / (1.0 + half_deltat_overtau_sigma_s);

                var r_xx_old: f32 = get_r_xx(x,y,z,_l);
                var r_yy_old: f32 = get_r_yy(x,y,z,_l);
                var r_zz_old: f32 = get_r_zz(x,y,z,_l);

                var r_xx: f32 = ((r_xx_old + div * deltat_phi_p - r_xx_old * half_deltat_overtau_sigma_p) * mult_factor_tau_sigma_p);
                var r_yy: f32 = ((r_yy_old + (vdvx_dx - div / 3.0) * deltat_phi_s - r_yy_old * half_deltat_overtau_sigma_s) * mult_factor_tau_sigma_s);
                var r_zz: f32 = ((r_zz_old + (vdvy_dy - div / 3.0) * deltat_phi_s - r_zz_old * half_deltat_overtau_sigma_s) * mult_factor_tau_sigma_s);

                sum_r_xx += r_xx_old + r_xx;
                sum_r_yy += r_yy_old + r_yy;
                sum_r_zz += r_zz_old + r_zz;

                set_r_xx_old(x,y,z,_l,r_xx_old);
                set_r_xx(x,y,z,_l,r_xx);
                set_r_yy_old(x,y,z,_l,r_yy_old);
                set_r_yy(x,y,z,_l,r_yy);
                set_r_zz_old(x,y,z,_l,r_zz_old);
                set_r_zz(x,y,z,_l,r_zz);
            }

            let lambda_23mu: f32 = lambda + 2.0 * mu / 3.0;

            sigmaxx = get_sigmaxx(x, y, z) + (lambdaplus2mu * vdvx_dx + lambda * (vdvy_dy + vdvz_dz) + lambda_23mu * 0.5 * sum_r_xx + mu * sum_r_yy) * dt;
            sigmayy = get_sigmayy(x, y, z) + (lambda * (vdvx_dx + vdvz_dz) + lambdaplus2mu * vdvy_dy + lambda_23mu * 0.5 * sum_r_xx + mu * sum_r_zz) * dt;
            sigmazz = get_sigmazz(x, y, z) + (lambda * (vdvx_dx + vdvy_dy) + lambdaplus2mu * vdvz_dz + lambdaplus2mu * 0.5 * sum_r_xx - (mu / 3.0)    * (sum_r_yy + sum_r_zz)) * dt;
        }
        else {
            sigmaxx = get_sigmaxx(x, y, z) + (lambdaplus2mu * vdvx_dx + lambda * (vdvy_dy + vdvz_dz))*dt;
            sigmayy = get_sigmayy(x, y, z) + (lambda        * (vdvx_dx + vdvz_dz) + lambdaplus2mu * vdvy_dy)*dt;
            sigmazz = get_sigmazz(x, y, z) + (lambda        * (vdvx_dx + vdvy_dy) + lambdaplus2mu * vdvz_dz)*dt;
        }
        set_sigmaxx(x, y, z, sigmaxx);
        set_sigmayy(x, y, z, sigmayy);
        set_sigmazz(x,y, z, sigmazz);
    }

    // Shear stresses
    // sigma_xy
    id_x_i = -get_idx_ff(last);
    id_x_f = sim_int_par.x_sz - get_idx_if(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    id_z_i = -get_idx_fh(last);
    id_z_f = sim_int_par.z_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdvy_dx: f32 = 0.0;
        var vdvx_dy: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvy_dx += get_fdc(c) * (get_vy(x + get_idx_if(c), y, z) - get_vy(x + get_idx_ff(c), y, z)) / dx;
            vdvx_dy += get_fdc(c) * (get_vx(x, y + get_idx_ih(c), z) - get_vx(x, y + get_idx_fh(c), z)) / dy;
        }

        let mdvy_dx_new: f32 = get_b_x(x - offset) * get_mdvy_dx(x, y, z) + get_a_x(x - offset) * vdvy_dx;
        let mdvx_dy_new: f32 = get_b_y_h(y - offset) * get_mdvx_dy(x, y, z) + get_a_y_h(y - offset) * vdvx_dy;

        vdvy_dx = vdvy_dx/get_k_x(x - offset)   + mdvy_dx_new;
        vdvx_dy = vdvx_dy/get_k_y_h(y - offset) + mdvx_dy_new;

        set_mdvy_dx(x, y, z, mdvy_dx_new);
        set_mdvx_dy(x, y, z, mdvx_dy_new);

        let rho = 0.25 * (get_rho(x + 1, y, z) + get_rho(x, y, z) + get_rho(x, y + 1, z) + get_rho(x + 1, y + 1, z));
        let cs = 0.25 * (get_cs(x + 1, y, z) + get_cs(x, y, z) + get_cs(x, y + 1, z) + get_cs(x + 1, y + 1, z));
        let mu: f32 = rho * (cs * cs);
        var sigmaxy: f32 = 0.0;

        if (visco_attn) {

            var sum_r_xy: f32 = 0.0;

            for (var _l: i32 = 0; _l < sim_int_par.n_sls; _l++) {
                var inv_tau_sigma_s: f32 = 1.0/get_tau(3,_l);
                var alpha_s:f32 =  get_tau(2,_l) / get_tau(3,_l);
                var deltat_phi_s: f32 = dt * (1.0 - alpha_s) * inv_tau_sigma_s / sum_alpha[1];

                var half_deltat_overtau_sigma_s: f32 = 0.5 * dt * inv_tau_sigma_s;
                var mult_factor_tau_sigma_s: f32 = 1.0 / (1.0 + half_deltat_overtau_sigma_s);

                var r_xy_old: f32 = get_r_xy(x,y,z,_l);
                var r_xy: f32 = ((r_xy_old + (vdvy_dx + vdvx_dy) * deltat_phi_s - r_xy_old * half_deltat_overtau_sigma_s) * mult_factor_tau_sigma_s);

                sum_r_xy += r_xy_old + r_xy;

                set_r_xy_old(x,y,z,_l,r_xy_old);
                set_r_xy(x,y,z,_l,r_xy);
        }

            sigmaxy = get_sigmaxy(x, y, z) + (mu * (vdvy_dx + vdvx_dy) + mu * 0.5 * sum_r_xy) * dt;
        }
        else {
            sigmaxy = get_sigmaxy(x, y, z) + (vdvx_dy + vdvy_dx) * mu * dt;
        }

        set_sigmaxy(x, y, z, sigmaxy);


    }

    // sigma_xz
    id_x_i = -get_idx_ff(last);
    id_x_f = sim_int_par.x_sz - get_idx_if(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    id_z_i = -get_idx_fh(last);
    id_z_f = sim_int_par.z_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdvz_dx: f32 = 0.0;
        var vdvx_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvz_dx += get_fdc(c) * (get_vz(x + get_idx_if(c), y, z) - get_vz(x + get_idx_ff(c), y, z)) / dx;
            vdvx_dz += get_fdc(c) * (get_vx(x, y, z + get_idx_ih(c)) - get_vx(x, y, z + get_idx_fh(c))) / dz;
        }

        var mdvz_dx_new: f32 = get_b_x(x - offset) * get_mdvz_dx(x, y, z) + get_a_x(x - offset) * vdvz_dx;
        var mdvx_dz_new: f32 = get_b_z_h(z - offset) * get_mdvx_dz(x, y, z) + get_a_z_h(z - offset) * vdvx_dz;

        vdvz_dx = vdvz_dx/get_k_x(x - offset)   + mdvz_dx_new;
        vdvx_dz = vdvx_dz/get_k_z_h(z - offset) + mdvx_dz_new;

        set_mdvz_dx(x, y, z, mdvz_dx_new);
        set_mdvx_dz(x, y, z, mdvx_dz_new);

        let rho = 0.25 * (get_rho(x + 1, y, z) + get_rho(x, y, z) + get_rho(x, y, z + 1) + get_rho(x + 1, y, z + 1));
        let cs = 0.25 * (get_cs(x + 1, y, z) + get_cs(x, y, z) + get_cs(x, y, z + 1) + get_cs(x + 1, y, z + 1));
        let mu: f32 = rho * (cs * cs);
        var sigmaxz: f32 = 0.0;
        if (visco_attn){

            var sum_r_xz: f32 = 0.0;
            for (var _l: i32 = 0; _l < sim_int_par.n_sls; _l++) {
                var inv_tau_sigma_s: f32 = 1.0 / get_tau(3, _l);
                var alpha_s: f32         = get_tau(2, _l) / get_tau(3, _l);
                var deltat_phi_s:              f32 = dt * (1.0 - alpha_s) * inv_tau_sigma_s / sum_alpha[1];
                var half_deltat_overtau_sigma_s: f32 = 0.5 * dt * inv_tau_sigma_s;
                var mult_factor_tau_sigma_s:   f32 = 1.0 / (1.0 + half_deltat_overtau_sigma_s);

                var r_xz_old: f32 = get_r_xz(x, y, z, _l);
                var r_xz: f32 = (r_xz_old + (vdvz_dx + vdvx_dz) * deltat_phi_s - r_xz_old * half_deltat_overtau_sigma_s) * mult_factor_tau_sigma_s;

                sum_r_xz += r_xz_old + r_xz;

                set_r_xz_old(x, y, z, _l, r_xz_old);
                set_r_xz(x, y, z, _l, r_xz);
            }

            sigmaxz = get_sigmaxz(x, y, z) + (mu * (vdvx_dz + vdvz_dx) + mu * 0.5 * sum_r_xz) * dt;

        }
        else{
            sigmaxz = get_sigmaxz(x, y, z) + (vdvx_dz + vdvz_dx) * mu * dt;
        }
        set_sigmaxz(x, y, z, sigmaxz);
    }


    // sigma_yz
    id_x_i = -get_idx_fh(last);
    id_x_f = sim_int_par.x_sz - get_idx_ih(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    id_z_i = -get_idx_fh(last);
    id_z_f = sim_int_par.z_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdvz_dy: f32 = 0.0;
        var vdvy_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvz_dy += get_fdc(c) * (get_vz(x, y + get_idx_ih(c), z) - get_vz(x, y + get_idx_fh(c), z)) / dy;
            vdvy_dz += get_fdc(c) * (get_vy(x, y, z + get_idx_ih(c)) - get_vy(x, y, z + get_idx_fh(c))) / dz;
        }

        var mdvz_dy_new: f32 = get_b_y_h(y - offset) * get_mdvz_dy(x, y, z) + get_a_y_h(y - offset) * vdvz_dy;
        var mdvy_dz_new: f32 = get_b_z_h(z - offset) * get_mdvy_dz(x, y, z) + get_a_z_h(z - offset) * vdvy_dz;

        vdvz_dy = vdvz_dy/get_k_y_h(y - offset) + mdvz_dy_new;
        vdvy_dz = vdvy_dz/get_k_z_h(z - offset) + mdvy_dz_new;

        set_mdvz_dy(x, y, z, mdvz_dy_new);
        set_mdvy_dz(x, y, z, mdvy_dz_new);

        let rho = 0.25 * (get_rho(x, y + 1, z) + get_rho(x, y, z) + get_rho(x, y, z + 1) + get_rho(x, y + 1, z + 1));
        let cs = 0.25 * (get_cs(x, y + 1, z) + get_cs(x, y, z) + get_cs(x, y, z + 1) + get_cs(x, y + 1, z + 1));
        let mu: f32 = rho * (cs * cs);
        var sigmayz: f32 = 0.0;
        
        if (visco_attn){
            var sum_r_yz: f32 = 0.0;

            for (var _l: i32 = 0; _l < sim_int_par.n_sls; _l++) {
                var inv_tau_sigma_s: f32 = 1.0 / get_tau(3, _l);
                var alpha_s: f32         = get_tau(2, _l) / get_tau(3, _l);

                var deltat_phi_s:              f32 = dt * (1.0 - alpha_s) * inv_tau_sigma_s / sum_alpha[1];
                var half_deltat_overtau_sigma_s: f32 = 0.5 * dt * inv_tau_sigma_s;
                var mult_factor_tau_sigma_s:   f32 = 1.0 / (1.0 + half_deltat_overtau_sigma_s);

                var r_yz_old: f32 = get_r_yz(x, y, z, _l);
                var r_yz: f32 = (r_yz_old + (vdvy_dz + vdvz_dy) * deltat_phi_s - r_yz_old * half_deltat_overtau_sigma_s) * mult_factor_tau_sigma_s;
                sum_r_yz += r_yz_old + r_yz;

                set_r_yz_old(x, y, z, _l, r_yz_old);
                set_r_yz(x, y, z, _l, r_yz);
            }

            sigmayz = get_sigmayz(x, y, z) + (mu * (vdvy_dz + vdvz_dy) + mu * 0.5 * sum_r_yz) * dt;
        }
        else{
            sigmayz = get_sigmayz(x, y, z) + (vdvy_dz + vdvz_dy) * mu * dt;
        }
        set_sigmayz(x, y, z, sigmayz);
    }

}

// Kernel to calculate velocities [vx, vy, vz]
@compute
@workgroup_size(wsx, wsy, wsz)
fn velocity_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // z thread index
    let dt: f32 = sim_flt_par.dt;
    let dx: f32 = sim_flt_par.dx;
    let dy: f32 = sim_flt_par.dy;
    let dz: f32 = sim_flt_par.dz;
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset: i32 = sim_int_par.fd_coeff - 1;

    // Vx
    var id_x_i: i32 = -get_idx_ff(last);    
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_if(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    var id_z_i: i32 = -get_idx_ff(last);
    var id_z_f: i32 = sim_int_par.z_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdsigmaxx_dx: f32 = 0.0;
        var vdsigmaxy_dy: f32 = 0.0;
        var vdsigmaxz_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdsigmaxx_dx += get_fdc(c) * (get_sigmaxx(x + get_idx_if(c), y, z) - get_sigmaxx(x + get_idx_ff(c), y, z)) / dx;
            vdsigmaxy_dy += get_fdc(c) * (get_sigmaxy(x, y + get_idx_if(c), z) - get_sigmaxy(x, y + get_idx_ff(c), z)) / dy;
            vdsigmaxz_dz += get_fdc(c) * (get_sigmaxz(x, y, z + get_idx_if(c)) - get_sigmaxz(x, y, z + get_idx_ff(c))) / dz;
        }

        var mdsxx_dx_new: f32 = get_b_x(x - offset) * get_mdsxx_dx(x, y, z) + get_a_x(x - offset) * vdsigmaxx_dx;
        var mdsxy_dy_new: f32 = get_b_y(y - offset) * get_mdsxy_dy(x, y, z) + get_a_y(y - offset) * vdsigmaxy_dy;
        var mdsxz_dz_new: f32 = get_b_z(z - offset) * get_mdsxz_dz(x, y, z) + get_a_z(z - offset) * vdsigmaxz_dz;

        vdsigmaxx_dx = vdsigmaxx_dx/get_k_x(x - offset) + mdsxx_dx_new;
        vdsigmaxy_dy = vdsigmaxy_dy/get_k_y(y - offset) + mdsxy_dy_new;
        vdsigmaxz_dz = vdsigmaxz_dz/get_k_z(z - offset) + mdsxz_dz_new;

        set_mdsxx_dx(x, y, z, mdsxx_dx_new);
        set_mdsxy_dy(x, y, z, mdsxy_dy_new);
        set_mdsxz_dz(x, y, z, mdsxz_dz_new);

        let rho: f32 = 0.5 * (get_rho(x + 1, y, z) + get_rho(x, y, z));
        if(rho > 0.0) {
            let vx: f32 = (vdsigmaxx_dx + vdsigmaxy_dy + vdsigmaxz_dz) * dt / rho + get_vx(x, y, z);
            set_vx(x, y, z, vx);
        }
    }

    // Vy
    id_x_i = -get_idx_fh(last);
    id_x_f = sim_int_par.x_sz - get_idx_ih(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    id_z_i = -get_idx_ff(last);
    id_z_f = sim_int_par.z_sz - get_idx_if(last);
    let rho: f32 = 0.5*(get_rho(x, y + 1, z) + get_rho(x, y, z));
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdsigmaxy_dx: f32 = 0.0;
        var vdsigmayy_dy: f32 = 0.0;
        var vdsigmayz_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdsigmaxy_dx += get_fdc(c) * (get_sigmaxy(x + get_idx_ih(c), y, z) - get_sigmaxy(x + get_idx_fh(c), y, z)) / dx;
            vdsigmayy_dy += get_fdc(c) * (get_sigmayy(x, y + get_idx_ih(c), z) - get_sigmayy(x, y + get_idx_fh(c), z)) / dy;
            vdsigmayz_dz += get_fdc(c) * (get_sigmayz(x, y, z + get_idx_if(c)) - get_sigmayz(x, y, z + get_idx_ff(c))) / dz;
        }

        var mdsxy_dx_new: f32 = get_b_x_h(x - offset) * get_mdsxy_dx(x, y, z) + get_a_x_h(x - offset) * vdsigmaxy_dx;
        var mdsyy_dy_new: f32 = get_b_y_h(y - offset) * get_mdsyy_dy(x, y, z) + get_a_y_h(y - offset) * vdsigmayy_dy;
        var mdsyz_dz_new: f32 = get_b_z(z - offset)   * get_mdsyz_dz(x, y, z) + get_a_z(z - offset)   * vdsigmayz_dz;

        vdsigmaxy_dx = vdsigmaxy_dx/get_k_x_h(x - offset) + mdsxy_dx_new;
        vdsigmayy_dy = vdsigmayy_dy/get_k_y_h(y - offset) + mdsyy_dy_new;
        vdsigmayz_dz = vdsigmayz_dz/get_k_z(z - offset)   + mdsyz_dz_new;

        set_mdsxy_dx(x, y, z, mdsxy_dx_new);
        set_mdsyy_dy(x, y, z, mdsyy_dy_new);
        set_mdsyz_dz(x, y, z, mdsyz_dz_new);

        if(rho > 0.0) {
            let vy: f32 = (vdsigmaxy_dx + vdsigmayy_dy + vdsigmayz_dz) * dt / rho + get_vy(x, y, z);
            set_vy(x, y, z, vy);
        }
    }
    
    // Vz
    id_x_i = -get_idx_fh(last);
    id_x_f = sim_int_par.x_sz - get_idx_ih(last);
    id_y_i = -get_idx_ff(last);
    id_y_f = sim_int_par.y_sz - get_idx_if(last);
    id_z_i = -get_idx_fh(last);
    id_z_f = sim_int_par.z_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdsigmaxz_dx: f32 = 0.0;
        var vdsigmayz_dy: f32 = 0.0;
        var vdsigmazz_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdsigmaxz_dx += get_fdc(c) * (get_sigmaxz(x + get_idx_ih(c), y, z) - get_sigmaxz(x + get_idx_fh(c), y, z)) / dx;
            vdsigmayz_dy += get_fdc(c) * (get_sigmayz(x, y + get_idx_if(c), z) - get_sigmayz(x, y + get_idx_ff(c), z)) / dy;
            vdsigmazz_dz += get_fdc(c) * (get_sigmazz(x, y, z + get_idx_ih(c)) - get_sigmazz(x, y, z + get_idx_fh(c))) / dz;
        }

        var mdsxz_dx_new: f32 = get_b_x_h(x - offset) * get_mdsxz_dx(x, y, z) + get_a_x_h(x - offset) * vdsigmaxz_dx;
        var mdsyz_dy_new: f32 = get_b_y(y - offset)   * get_mdsyz_dy(x, y, z) + get_a_y(y - offset)   * vdsigmayz_dy;
        var mdszz_dz_new: f32 = get_b_z_h(z - offset) * get_mdszz_dz(x, y, z) + get_a_z_h(z - offset) * vdsigmazz_dz;

        vdsigmaxz_dx = vdsigmaxz_dx/get_k_x_h(x - offset) + mdsxz_dx_new;
        vdsigmayz_dy = vdsigmayz_dy/get_k_y(y - offset)   + mdsyz_dy_new;
        vdsigmazz_dz = vdsigmazz_dz/get_k_z_h(z - offset) + mdszz_dz_new;

        set_mdsxz_dx(x, y, z, mdsxz_dx_new);
        set_mdsyz_dy(x, y, z, mdsyz_dy_new);
        set_mdszz_dz(x, y, z, mdszz_dz_new);

        if(rho > 0.0) {
            let vz: f32 = (vdsigmaxz_dx + vdsigmayz_dy + vdsigmazz_dz) * dt / rho + get_vz(x, y, z);
            set_vz(x, y, z, vz);
        }
    }
    
    else {
        set_vx(x,y,z, 0.0);
        set_vy(x,y,z, 0.0);
        set_vz(x,y,z, 0.0);
    }

    // Add the source force
    let idx_src_term: i32 = get_idx_source_term(x, y, z);
    if(idx_src_term != -1 && rho > 0.0) {
        let val_src: f32 = get_source_term(sim_int_par.it, idx_src_term);
        let vz: f32 = get_vz(x, y, z) +  val_src * dt / rho;
        set_vz(x, y, z, vz);
    }

    // Compute velocity norm L2
    let v_2_old: f32 = v_2;
    let v2: f32 = get_vx(x, y, z) * get_vx(x, y, z) + get_vy(x, y, z) * get_vy(x, y, z) + get_vz(x, y, z) * get_vz(x, y, z); ;
    v_2 = max(v_2_old, v2);

}

// Kernel to add the sources forces
@compute
@workgroup_size(wsx, wsy, wsz)
fn sources_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // z thread index
    let dt: f32 = sim_flt_par.dt;
    let it: i32 = sim_int_par.it;

    // Add the source force
    let idx_src_term: i32 = get_idx_source_term(x, y, z);
    let rho: f32 = 0.25 * (get_rho(x, y, z) + get_rho(x + 1, y, z) + get_rho(x + 1, y + 1, z) + get_rho(x, y + 1, z));
    if(idx_src_term != -1 && rho > 0.0) {
        let val_src: f32 = get_source_term(it, idx_src_term);
        let vy: f32 = get_vy(x, y, z) +  val_src * dt / rho;
        set_vy(x, y, z, vy);
    }
}

// Kernel to finish iteration term
@compute
@workgroup_size(wsx, wsy, wsz)
fn finish_it_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // y thread index
    let last: i32 = sim_int_par.fd_coeff - 1;
    let id_x_i: i32 = -get_idx_fh(last);
    let id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    let id_y_i: i32 = -get_idx_fh(last);
    let id_y_f: i32 = sim_int_par.y_sz - get_idx_ih(last);
    let id_z_i: i32 = -get_idx_fh(last);
    let id_z_f: i32 = sim_int_par.z_sz - get_idx_ih(last);
    let v_2_old: f32 = v_2;

    // Apply Dirichlet conditions
    if(x <= id_x_i || x >= id_x_f || y <= id_y_i || y >= id_y_f || z <= id_z_i || z >= id_z_f) {
        set_vx(x,y,z, 0.0);
        set_vy(x,y,z, 0.0);
        set_vz(x,y,z, 0.0);
    }

    // Compute velocity norm L2
    let v2: f32 = get_vx(x, y, z) * get_vx(x, y, z) + get_vy(x, y, z) * get_vy(x, y, z) + get_vz(x, y, z) * get_vz(x, y, z); 
    v_2 = max(v_2_old, v2);
}

// Kernel to store sensors velocity
@compute
@workgroup_size(idx_rec_offset, 1, 1)
fn store_sensors_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let sensor: i32 = i32(index.x);
    
    if (sensor >= sim_int_par.n_rec_el) { return; }

    let it: i32 = sim_int_par.it;

    for(var pt: i32 = get_offset_sensor(sensor); get_idx_sensor(pt) == sensor; pt++) {
        if(it >= get_delay_rec(sensor)) {
            let x: i32 = get_idx_x_sensor(pt);
            let y: i32 = get_idx_y_sensor(pt);
            let z: i32 = get_idx_z_sensor(pt);

            let value_sens_sigzz: f32 = get_sens_sigzz(it, sensor) + get_sigmazz(x, y, z);
            set_sens_sigzz(it, sensor, value_sens_sigzz);
        }
    }
}

// Kernel to increase time iteraction [it]
@compute
@workgroup_size(1)
fn incr_it_kernel() {
    sim_int_par.it += 1;
}