#include "sim_cuda_cpp.hpp"

#include <json/json.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kFloatEpsilon = std::numeric_limits<float>::epsilon();

float round_decimal(float value, int decimals) {
    const float scale = std::pow(10.0f, static_cast<float>(decimals));
    return std::nearbyint(value * scale) / scale;
}

int decimal_places(float step) {
    if (step <= 0.0f) throw std::invalid_argument("ROI step precisa ser positivo");
    return static_cast<int>(std::fabs(std::log10(step))) + 2;
}

float number(const Json::Value& object, const char* key, float fallback) {
    if (!object.isMember(key) || object[key].isNull()) return fallback;
    const auto& value = object[key];
    if (value.isNumeric()) return value.asFloat();
    if (value.isString()) return std::stof(value.asString());
    throw std::invalid_argument(std::string("Valor numerico invalido em '") + key + "'");
}

int integer(const Json::Value& object, const char* key, int fallback) {
    return static_cast<int>(number(object, key, static_cast<float>(fallback)));
}

std::string text(const Json::Value& object, const char* key, std::string fallback = {}) {
    if (!object.isMember(key) || object[key].isNull()) return fallback;
    if (object[key].isString()) return object[key].asString();
    return object[key].asString();
}

bool boolean_value(const Json::Value& value, bool fallback = false) {
    if (value.isBool()) return value.asBool();
    if (value.isNumeric()) return value.asInt() != 0;
    if (value.isString()) {
        std::string s = value.asString();
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "true" || s == "1" || s == "yes") return true;
        if (s == "false" || s == "0" || s == "no") return false;
    }
    return fallback;
}

bool flag(const Json::Value& object, const char* key, bool fallback = false) {
    return object.isMember(key) ? boolean_value(object[key], fallback) : fallback;
}

std::array<float, 3> vector3(const Json::Value& value, std::array<float, 3> fallback = {0, 0, 0}) {
    if (!value.isArray()) return fallback;
    const Json::Value& row = !value.empty() && value[0].isArray() ? value[0] : value;
    for (Json::ArrayIndex i = 0; i < 3 && i < row.size(); ++i) {
        if (row[i].isNumeric()) fallback[i] = row[i].asFloat();
        else if (row[i].isString()) fallback[i] = std::stof(row[i].asString());
    }
    return fallback;
}

struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

std::size_t product(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
}

NpyArray load_npy(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Nao foi possivel abrir NPY: " + path.string());
    std::array<unsigned char, 8> preamble{};
    file.read(reinterpret_cast<char*>(preamble.data()), preamble.size());
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (!file || !std::equal(std::begin(magic), std::end(magic), preamble.begin())) {
        throw std::runtime_error("Cabecalho NPY invalido: " + path.string());
    }
    std::uint32_t header_size = 0;
    if (preamble[6] == 1) {
        std::uint16_t short_size = 0;
        file.read(reinterpret_cast<char*>(&short_size), sizeof(short_size));
        header_size = short_size;
    } else if (preamble[6] == 2 || preamble[6] == 3) {
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    } else {
        throw std::runtime_error("Versao NPY nao suportada");
    }
    if (header_size > 1024 * 1024) throw std::runtime_error("Cabecalho NPY excessivamente grande");
    std::string header(header_size, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!file) throw std::runtime_error("Cabecalho NPY truncado");

    std::smatch match;
    if (!std::regex_search(header, match, std::regex("'descr'\\s*:\\s*'([^']+)'"))) {
        throw std::runtime_error("Descritor dtype ausente em NPY");
    }
    const std::string dtype = match[1].str();
    if (dtype != "<f4" && dtype != "=f4" && dtype != "|f4" && dtype != "<f8" && dtype != "=f8") {
        throw std::runtime_error("NPY deve ser float32/float64; descritor recebido: " + dtype);
    }
    const bool fortran = std::regex_search(header, match, std::regex("'fortran_order'\\s*:\\s*(True|False)")) &&
                         match[1].str() == "True";
    if (fortran) throw std::runtime_error("NPY Fortran-order nao e suportado; esperado C-order");
    if (!std::regex_search(header, match, std::regex("'shape'\\s*:\\s*\\(([^)]*)\\)"))) {
        throw std::runtime_error("Shape ausente em NPY");
    }
    NpyArray array;
    std::stringstream dims(match[1].str());
    std::string part;
    while (std::getline(dims, part, ',')) {
        const auto first = part.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        const auto last = part.find_last_not_of(" \t");
        array.shape.push_back(static_cast<std::size_t>(std::stoull(part.substr(first, last - first + 1))));
    }
    if (array.shape.empty()) throw std::runtime_error("Shape NPY vazio");
    const std::size_t count = product(array.shape);
    array.values.resize(count);
    if (dtype.find("f8") != std::string::npos) {
        std::vector<double> tmp(count);
        file.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(count * sizeof(double)));
        if (!file) throw std::runtime_error("Dados NPY truncados");
        std::transform(tmp.begin(), tmp.end(), array.values.begin(), [](double v) { return static_cast<float>(v); });
    } else {
        file.read(reinterpret_cast<char*>(array.values.data()), static_cast<std::streamsize>(count * sizeof(float)));
        if (!file) throw std::runtime_error("Dados NPY truncados");
    }
    return array;
}

void save_npy(const fs::path& path, const std::vector<float>& values, const std::vector<std::size_t>& shape) {
    if (product(shape) != values.size()) throw std::invalid_argument("Shape NPY nao coincide com o tamanho dos dados");
    std::ostringstream shape_text;
    shape_text << '(';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) shape_text << ", ";
        shape_text << shape[i];
    }
    if (shape.size() == 1) shape_text << ',';
    shape_text << ')';
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_text.str() + ", }";
    const std::size_t padding = (16 - ((10 + header.size() + 1) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<std::uint16_t>::max()) throw std::runtime_error("Header NPY muito grande");

    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Nao foi possivel escrever NPY: " + path.string());
    const unsigned char preamble[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const auto header_len = static_cast<std::uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(preamble), sizeof(preamble));
    file.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) throw std::runtime_error("Erro ao salvar NPY: " + path.string());
}

struct PmlProfile {
    std::vector<float> a, b, k;
};

struct Roi {
    float x0 = 0, y0 = 0, z0 = 0;
    float width = 0, height = 0, depth = 0;
    int w_len = 0, h_len = 0, d_len = 1;
    int pml_xmin = 0, pml_xmax = 0, pml_zmin = 0, pml_zmax = 0, pad = 0;
    float dx = 0, dz = 0, rcoef = 1e-4f, npower = 2, kmax = 1;
    int dec_x = 2, dec_z = 2;
        int dec_y = 2;
    std::vector<float> w_points, h_points;

    Roi(const Json::Value& config, int padding, const NpyArray* rho_map) : pad(padding) {
        const auto coords = vector3(config.get("coord_ref", Json::Value(Json::arrayValue)));
        x0 = coords[0]; y0 = coords[1]; z0 = coords[2];
        width = number(config, "width", 30.0f);
        height = number(config, "height", 30.0f);
        depth = number(config, "depth", 0.0f);
        w_len = integer(config, "w_len", 300);
        h_len = integer(config, "h_len", 300);
        d_len = integer(config, "d_len", 1);
        const float dx0 = width / w_len;
        const float dz0 = height / h_len;
        dx = round_decimal(dx0, decimal_places(dx0));
        dz = round_decimal(dz0, decimal_places(dz0));
        const float dy0 = depth > 0.0f && d_len > 1 ? depth / d_len : dx;
        dec_y = decimal_places(dy0);
        const float original_width = width;
        const float original_height = height;
        bool width_expanded = false, height_expanded = false;
        if (rho_map && rho_map->shape.size() == 2) {
            if (static_cast<int>(rho_map->shape[0]) > w_len) {
                w_len = static_cast<int>(rho_map->shape[0]); width_expanded = true;
            }
            if (static_cast<int>(rho_map->shape[1]) > h_len) {
                h_len = static_cast<int>(rho_map->shape[1]); height_expanded = true;
            }
        }
        width = width_expanded ? dx * w_len : original_width;
        height = height_expanded ? dz * h_len : original_height;
        pml_xmin = integer(config, "len_pml_xmin", 10);
        pml_xmax = integer(config, "len_pml_xmax", 10);
        pml_zmin = integer(config, "len_pml_zmin", 10);
        pml_zmax = integer(config, "len_pml_zmax", 10);
        rcoef = number(config, "rcoef_pml", 0.0001f);
        npower = number(config, "npower_pml", 2.0f);
        kmax = number(config, "k_max_pml", 1.0f);
        dec_x = decimal_places(dx);
        dec_z = decimal_places(dz);
        w_points.resize(w_len);
        h_points.resize(h_len);
        for (int i = 0; i < w_len; ++i) w_points[i] = round_decimal(x0 + width * i / w_len, dec_x);
        for (int i = 0; i < h_len; ++i) h_points[i] = round_decimal(z0 + height * i / h_len, dec_z);
        if (w_len <= 0 || h_len <= 0 || dx <= 0 || dz <= 0 || pml_xmin < 0 || pml_xmax < 0 ||
            pml_zmin < 0 || pml_zmax < 0 || pad < 0) {
            throw std::invalid_argument("Geometria da ROI invalida");
        }
    }

    int nx() const { return w_len + pml_xmin + pml_xmax + 2 * pad; }
    int ny() const { return h_len + pml_zmin + pml_zmax + 2 * pad; }
    int ix_min() const { return pml_xmin + pad; }
    int ix_max() const { return w_len + pml_xmin + pad; }
    int iy_min() const { return pml_zmin + pad; }
    int iy_max() const { return h_len + pml_zmin + pad; }

    std::array<int, 2> nearest(float x, float y, float z) const {
        const float x_last = w_points.back();
        const float z_last = h_points.back();
        if (x < round_decimal(x0, dec_x) || x > x_last || y < round_decimal(y0, dec_y) || y > round_decimal(y0, dec_y) ||
            z < round_decimal(z0, dec_z) || z > z_last) {
            std::ostringstream message;
            message << "Ponto [" << x << ", " << y << ", " << z << "] fora da ROI";
            throw std::out_of_range(message.str());
        }
        const float x_target = round_decimal(x - dx / std::pow(10.0f, dec_x - 1), dec_x);
        const float z_target = round_decimal(z - dz / std::pow(10.0f, dec_z - 1), dec_z);
        int ix = 0, iz = 0;
        float best_x = std::numeric_limits<float>::max();
        float best_z = std::numeric_limits<float>::max();
        for (int i = 0; i < w_len; ++i) {
            const float point = w_points[i];
            if (std::fabs(point - x_target) < best_x) { best_x = std::fabs(point - x_target); ix = i; }
        }
        for (int i = 0; i < h_len; ++i) {
            const float point = h_points[i];
            if (std::fabs(point - z_target) < best_z) { best_z = std::fabs(point - z_target); iz = i; }
        }
        return {ix + pml_xmin + pad, iz + pml_zmin + pad};
    }

    PmlProfile pml(char axis, bool half, float dt, float cp, float alpha_max) const {
        const bool x_axis = axis == 'x';
        const int len = x_axis ? w_len : h_len;
        const int pmin = x_axis ? pml_xmin : pml_zmin;
        const int pmax = x_axis ? pml_xmax : pml_zmax;
        const float delta = x_axis ? dx : dz;
        const int decimals = x_axis ? dec_x : dec_z;
        const int total = len + pmin + pmax;
        const float thickness_left = pmin * delta;
        const float thickness_right = pmax * delta;
        const float thickness = thickness_left + thickness_right;
        PmlProfile out;
        out.a.assign(total, 0.0f);
        out.b.assign(total, 1.0f);
        out.k.assign(total, 1.0f);
        if (thickness <= 0.0f) return out;
        const float orig_left = pmin * delta;
        const float orig_right = (pmin + len - 1) * delta;
        const float d0 = -(npower + 1.0f) * cp * std::log(rcoef) / thickness;
        for (int i = 0; i < total; ++i) {
            const float val = round_decimal(delta * i, decimals);
            const float shift = half ? delta / 2.0f : 0.0f;
            const float left = round_decimal(orig_left - (val + shift), decimals);
            const float right = round_decimal((val + shift) - orig_right, decimals);
            const bool in_left = thickness_left > 0.0f && left >= 0.0f;
            const bool in_right = thickness_right > 0.0f && right >= 0.0f;
            float p = 0.0f;
            if (in_left) p = left / thickness_left;
            if (in_right) p = right / thickness_right;
            const bool mask = in_left || in_right;
            const float d = d0 * std::pow(p, npower);
            const float k = 1.0f + (kmax - 1.0f) * std::pow(p, npower);
            const float alpha = alpha_max * (1.0f - (mask ? p : 1.0f));
            const float b = std::exp(-(d / k + alpha) * dt);
            const float a = d > 1e-6f ? d * (b - 1.0f) / (k * (d + k * alpha)) : 0.0f;
            out.a[i] = a; out.b[i] = b; out.k[i] = k;
        }
        return out;
    }
};

struct Element {
    bool tx = false, rx = false;
    bool point = false;
    std::array<float, 3> center{0, 0, 0};
    float dim_a = 0.5f, dim_p = 10.0f, freq = 5.0f, bw = 0.5f, gain = 1.0f;
    float tx_delay = 0.0f, rx_delay = 0.0f;
    std::vector<int> cells;
    int tx_channel = -1, rx_channel = -1;
};

struct Probe {
    bool linear = false;
    std::string id;
    std::array<float, 3> center{0, 0, 0};
    int num_elem = 1;
    float dim_a = 0.5f, dim_p = 10.0f, inter_elem = 0.1f;
    float freq = 5.0f, bw = 0.5f, gain = 1.0f;
    std::vector<Element> elements;
};

std::vector<bool> channel_flags(const Json::Value& value, int count, bool default_value) {
    std::vector<bool> out(count, default_value);
    if (value.isNull()) return out;
    if (value.isString()) {
        const std::string mode = value.asString();
        if (mode == "all") return std::vector<bool>(count, true);
        if (mode == "none") return std::vector<bool>(count, false);
        throw std::invalid_argument("Flag de canal deve ser all, none ou lista");
    }
    if (!value.isArray()) throw std::invalid_argument("Flag de canal invalida");
    std::fill(out.begin(), out.end(), false);
    for (Json::ArrayIndex i = 0; i < value.size() && i < static_cast<Json::ArrayIndex>(count); ++i) {
        out[i] = boolean_value(value[i]);
    }
    return out;
}

std::vector<float> channel_delays(const Json::Value& value, int count) {
    std::vector<float> out(count, 0.0f);
    if (value.isNull()) return out;
    if (value.isNumeric() || value.isString()) {
        const float d = value.isNumeric() ? value.asFloat() : std::stof(value.asString());
        std::fill(out.begin(), out.end(), d);
    } else if (value.isArray()) {
        for (Json::ArrayIndex i = 0; i < value.size() && i < static_cast<Json::ArrayIndex>(count); ++i) {
            out[i] = value[i].isNumeric() ? value[i].asFloat() : std::stof(value[i].asString());
        }
    } else {
        throw std::invalid_argument("Atraso de canal invalido");
    }
    return out;
}

std::vector<int> element_cells(const Roi& roi, const Probe& probe, const Element& element) {
    if (element.point) {
        const auto idx = roi.nearest(element.center[0], element.center[1], element.center[2]);
        return {idx[0] * roi.ny() + idx[1]};
    }
    const int dec_w = roi.dec_x;
    const int npoints = std::max(0, static_cast<int>(round_decimal(element.dim_a / roi.dx, dec_w) + 0.5f));
    const float pitch = probe.dim_a + probe.inter_elem;
    const float offset = ((probe.num_elem - 1) * pitch + probe.dim_a) / 2.0f;
    const float local_center_x = round_decimal(probe.dim_a / 2.0f + (&element - probe.elements.data()) * pitch - offset,
                                               std::max(roi.dec_x, roi.dec_z));
    const bool multiple = (npoints / 2) != 0;
    const float first_offset = multiple ? (element.dim_a - roi.dx) / 2.0f : element.dim_a / 2.0f;
    const float x_coord = local_center_x - first_offset + probe.center[0];
    const float y_coord = probe.center[1];
    const float z_coord = probe.center[2];
    const auto start = roi.nearest(x_coord, y_coord, z_coord);
    std::vector<int> points;
    points.reserve(npoints);
    for (int p = 0; p < npoints; ++p) {
        const int x = start[0] + p;
        const int y = start[1];
        if (x < 0 || x >= roi.nx() || y < 0 || y >= roi.ny()) {
            throw std::out_of_range("Pontos do elemento linear excedem a grade");
        }
        points.push_back(x * roi.ny() + y);
    }
    return points;
}

Probe parse_probe(const Json::Value& node, const Roi& roi) {
    Probe probe;
    probe.linear = node.isMember("linear");
    if (!probe.linear && !node.isMember("point")) throw std::invalid_argument("Probe deve ser linear ou point");
    const Json::Value& cfg = probe.linear ? node["linear"] : node["point"];
    probe.id = text(cfg, "id", "");
    probe.center = vector3(cfg.get("coord_center", Json::Value(Json::arrayValue)));
    probe.freq = number(cfg, "freq", 5.0f);
    probe.bw = number(cfg, "bw", 0.5f);
    probe.gain = number(cfg, "gain", 1.0f);

    if (probe.linear) {
        probe.num_elem = integer(cfg, "num_elem", 32);
        probe.dim_a = number(cfg, "dim_a", 0.5f);
        probe.dim_p = number(cfg, "dim_p", 10.0f);
        probe.inter_elem = number(cfg, "inter_elem", 0.1f);
        const auto emitters = channel_flags(cfg.get("emitters", Json::Value("all")), probe.num_elem, true);
        const auto receivers = channel_flags(cfg.get("receivers", Json::Value("all")), probe.num_elem, true);
        const auto tx_delays = channel_delays(cfg.get("t0_emission", Json::Value()), probe.num_elem);
        const auto rx_delays = channel_delays(cfg.get("t0_reception", Json::Value()), probe.num_elem);
        probe.elements.resize(probe.num_elem);
        for (int i = 0; i < probe.num_elem; ++i) {
            auto& e = probe.elements[i];
            e.tx = emitters[i]; e.rx = receivers[i]; e.dim_a = probe.dim_a; e.dim_p = probe.dim_p;
            e.freq = probe.freq; e.bw = probe.bw; e.gain = probe.gain;
            e.tx_delay = tx_delays[i]; e.rx_delay = rx_delays[i];
        }
    } else {
        probe.num_elem = 1;
        auto& e = probe.elements.emplace_back();
        e.point = true;
        e.center = probe.center;
        e.tx = boolean_value(cfg.get("emitter", Json::Value("True")), true);
        e.rx = boolean_value(cfg.get("receiver", Json::Value("False")), false);
        e.freq = probe.freq; e.bw = probe.bw; e.gain = probe.gain;
        e.tx_delay = number(cfg, "t0_emission", 0.0f);
        e.rx_delay = number(cfg, "t0_reception", 0.0f);
    }

    for (auto& e : probe.elements) {
        if ((!probe.linear && (e.tx || e.rx)) || (probe.linear && (e.tx || e.rx))) {
            try {
                e.cells = element_cells(roi, probe, e);
            } catch (const std::out_of_range&) {
                if (!probe.linear) throw;
                e.cells.clear();
            }
        }
    }
    return probe;
}

struct Waveform {
    std::vector<float> pressure;
};

float gaussian_a(float fc, float bw, float bwr = -6.0f) {
    if (fc < 0.0f || bw <= 0.0f || bwr >= 0.0f) throw std::invalid_argument("Parametros gaussianos invalidos");
    const float ref = std::pow(10.0f, bwr / 20.0f);
    return -std::pow(kPi * fc * bw, 2.0f) / (4.0f * std::log(ref));
}

void pulse_components(float t, float fc, float bw, float& pulse, float& envelope) {
    const float a = gaussian_a(fc, bw);
    const float phase = 2.0f * kPi * fc * t;
    const float gaussian = std::exp(-a * t * t);
    envelope = gaussian;
    pulse = gaussian * std::cos(phase);
}

std::vector<float> make_source(const Element& element, int samples, float dt, int ord_der,
                               bool source_env, bool linear_probe) {
    std::vector<float> raw(samples, 0.0f);
    if (!element.tx) return raw;
    const int time_decimals = decimal_places(dt);
    for (int i = 0; i < samples; ++i) {
        float t = static_cast<float>(i) * dt;
        if (linear_probe) t = round_decimal(t, time_decimals);
        t -= element.tx_delay;
        const float a = gaussian_a(element.freq, element.bw);
        const float phase = 2.0f * kPi * element.freq * t;
        const float gaussian = std::exp(-a * t * t);
        float pulse = 0.0f, envelope = gaussian;
        if (linear_probe) {
            pulse = gaussian * std::cos(phase);
        } else if (ord_der == 2) {
            const float at = a * t;
            const float q = 2.0f * a * t * t - 1.0f;
            envelope = 2.0f * a * gaussian * q;
            pulse = gaussian * ((2.0f * a * q - 4.0f * std::pow(kPi * element.freq, 2.0f)) * std::cos(phase) +
                                8.0f * kPi * a * element.freq * t * std::sin(phase));
        } else if (ord_der == 1) {
            envelope = -2.0f * gaussian * (a * t);
            pulse = -2.0f * gaussian * (a * t * std::cos(phase) + kPi * element.freq * std::sin(phase));
        } else {
            pulse = gaussian * std::cos(phase);
        }
        float sample = source_env ? envelope : pulse;
        if (std::fabs(sample) < kFloatEpsilon) sample = 0.0f;
        raw[i] = sample;
    }

    if (!linear_probe) {
        for (auto& value : raw) value *= element.gain;
        return raw;
    }
    if (samples < 2) return std::vector<float>(samples, 0.0f);
    const float sample_dt = round_decimal(dt, time_decimals);
    std::vector<float> integrated(samples, 0.0f), differentiated(samples, 0.0f);
    for (int i = 0; i < samples; ++i) integrated[i] = element.gain * raw[i] / sample_dt;
    for (int i = 0; i + 1 < samples; ++i) differentiated[i] = integrated[i + 1] - integrated[i];
    differentiated.back() = -integrated.back();
    return differentiated;
}

struct LawSet {
    std::vector<std::vector<float>> delay;
    std::vector<std::vector<float>> amplitude;
};

LawSet load_law(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Nao foi possivel abrir lei de emissao: " + path.string());
    std::string line;
    std::getline(file, line); std::getline(file, line); std::getline(file, line);
    std::vector<std::array<float, 9>> rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream stream(line);
        std::array<float, 9> row{};
        for (int col = 0; col < 9; ++col) {
            std::string token;
            if (!std::getline(stream, token, '\t')) {
                if (col == 8) break;
                throw std::runtime_error("Linha invalida no arquivo .law");
            }
            row[col] = std::stof(token);
        }
        rows.push_back(row);
    }
    if (rows.empty()) throw std::runtime_error("Arquivo .law sem dados");
    int elements = 0;
    int max_shot_t = 0, max_shot_s = 0;
    for (const auto& r : rows) {
        elements = std::max(elements, static_cast<int>(r[4]));
        max_shot_s = std::max(max_shot_s, static_cast<int>(r[1]));
        max_shot_t = std::max(max_shot_t, static_cast<int>(r[2]));
    }
    const int shots = max_shot_s != 0 ? max_shot_s + 1 : max_shot_t + 1;
    LawSet law;
    law.delay.assign(shots, std::vector<float>(elements, 0.0f));
    law.amplitude.assign(shots, std::vector<float>(elements, 0.0f));
    for (int shot = 0; shot < shots; ++shot) {
        for (int channel = 0; channel < elements; ++channel) {
            const std::size_t row_id = static_cast<std::size_t>(shot) * elements + channel;
            if (row_id >= rows.size()) throw std::runtime_error("Arquivo .law incompleto para os shots declarados");
            law.delay[shot][channel] = rows[row_id][5];
            law.amplitude[shot][channel] = rows[row_id][6];
        }
    }
    return law;
}

struct Prepared {
    Json::Value config;
    Roi roi;
    int acc = 2, n_steps = 1000, it_display = 10, n_iter = 1;
    float dt = 1.0f, rho = 7800.0f, cp = 5.9f, cs = 3.23f, cp_max = 5.9f;
    std::vector<float> cs_map;
    bool show_debug = false, show_anim = false, show_figs = false;
    bool plot_results = false, plot_error = false, plot_sensors = false, plot_bscan = false;
    bool save_results = false, save_field = false, save_sensors = false, save_bscan = false, save_sources = false;
    bool source_env = false;
    float min_fields = -100.0f, max_fields = 100.0f;
    std::string sim_model = "split", name = "CuPy-CUDA-C++";
    fs::path results_dir = ".";
    std::vector<float> coefficients;
    std::vector<float> rho_grid, cp_grid;
    std::vector<Probe> probes;
    std::vector<int> source_map, sensor_map, receiver_delays;
    std::vector<std::pair<int, int>> tx_channels;
    std::vector<std::pair<int, int>> rx_channels;
    std::vector<float> source_term;
    std::array<PmlProfile, 6> pml;
    std::optional<LawSet> laws;

    Prepared(const Json::Value& root, const std::string& model)
        : config(root), roi(make_roi(root, model)) {
        sim_model = model;
        const Json::Value simul = root.get("simul_params", Json::Value(Json::objectValue));
        acc = integer(simul, "acc", 2);
        if (acc > 6) {
            std::cerr << "Acuracia nao suportada; usando 6\n";
            acc = 6;
        }
        if (acc < 2) throw std::invalid_argument("Acuracia minima suportada: 2");
        const int pad_expected = acc - 1;
        if (roi.pad != pad_expected) throw std::logic_error("Padding interno inconsistente");
        n_steps = integer(simul, "time_steps", 1000);
        dt = number(simul, "dt", 1.0f);
        it_display = integer(simul, "it_display", 10);
        if (n_steps <= 0 || dt <= 0 || it_display <= 0) throw std::invalid_argument("Parametros de tempo invalidos");
        rho = number(root.get("specimen_params", Json::Value(Json::objectValue)), "rho", 7800.0f);
        cp = number(root.get("specimen_params", Json::Value(Json::objectValue)), "cp", 5.9f);
        const Json::Value specimen = root.get("specimen_params", Json::Value(Json::objectValue));
        cs = number(specimen, "cs", 3.23f);
        if (specimen.isMember("cs_map")) cs_map = load_npy(text(specimen, "cs_map")).values;
        build_stencil();
        build_material_maps(root);
        build_probes(root);
        build_pml();
        read_run_options(root);
        build_default_sources();
        load_optional_law(root);
    }

    static Roi make_roi(const Json::Value& root, const std::string&) {
        const Json::Value specimen = root.get("specimen_params", Json::Value(Json::objectValue));
        std::optional<NpyArray> rho_map;
        if (specimen.isMember("rho_map")) rho_map = load_npy(text(specimen, "rho_map"));
        const Json::Value simul = root.get("simul_params", Json::Value(Json::objectValue));
        int acc = integer(simul, "acc", 2);
        if (acc > 6) acc = 6;
        if (acc < 2) acc = 2;
        return Roi(root.get("roi", Json::Value(Json::objectValue)), acc - 1, rho_map ? &*rho_map : nullptr);
    }

    void build_stencil() {
        static const std::vector<std::vector<float>> lui = {
            {9.0f/8.0f, -1.0f/24.0f},
            {75.0f/64.0f, -25.0f/384.0f, 3.0f/640.0f},
            {1225.0f/1024.0f, -245.0f/3072.0f, 49.0f/5120.0f, -5.0f/7168.0f},
            {19845.0f/16384.0f, -735.0f/8192.0f, 567.0f/40960.0f, -405.0f/229376.0f, 35.0f/294912.0f},
            {160083.0f/131072.0f, -12705.0f/131072.0f, 22869.0f/1310720.0f, -5445.0f/1835008.0f, 847.0f/2359296.0f, -63.0f/2883584.0f}
        };
        if (sim_model != "split") throw std::invalid_argument("O backend SimulatorCupyCuda implementa o modelo split");
        coefficients = lui[std::min(acc - 2, 4)];
    }

    void build_material_maps(const Json::Value& root) {
        const std::size_t cells = static_cast<std::size_t>(roi.nx()) * roi.ny();
        rho_grid.assign(cells, rho);
        cp_grid.assign(cells, cp);
        const Json::Value specimen = root.get("specimen_params", Json::Value(Json::objectValue));
        auto apply_map = [&](const char* key, std::vector<float>& grid) {
            if (!specimen.isMember(key)) return;
            const auto map = load_npy(text(specimen, key));
            if (map.shape.size() != 2) throw std::invalid_argument(std::string(key) + " deve ser 2D");
            const int mx = static_cast<int>(map.shape[0]), my = static_cast<int>(map.shape[1]);
            if (mx < roi.nx() && my < roi.ny()) {
                if (mx != roi.w_len || my != roi.h_len) {
                    throw std::invalid_argument(std::string(key) + " menor que grade e incompatível com ROI física");
                }
                for (int x = 0; x < mx; ++x) for (int y = 0; y < my; ++y) {
                    grid[(x + roi.ix_min()) * roi.ny() + y + roi.iy_min()] = map.values[x * my + y];
                }
            } else if (mx > roi.nx() && my > roi.ny()) {
                for (int x = 0; x < roi.nx(); ++x) for (int y = 0; y < roi.ny(); ++y) {
                    grid[x * roi.ny() + y] = map.values[x * my + y];
                }
            } else if (mx == roi.nx() && my == roi.ny()) {
                grid = map.values;
            } else {
                throw std::invalid_argument(std::string(key) + " tem dimensoes incompativeis com ROI");
            }
        };
        apply_map("rho_map", rho_grid);
        // Preserva o aliasing do Simulator Python: a media staggered e escrita no mapa vx.
        const std::vector<float> rho_original = rho_grid;
        for (int x = 0; x + 1 < roi.nx(); ++x) {
            for (int y = 0; y + 1 < roi.ny(); ++y) {
                const auto i = static_cast<std::size_t>(x) * roi.ny() + y;
                rho_grid[i] = 0.25f * (rho_original[i] + rho_original[i + roi.ny()] +
                                       rho_original[i + roi.ny() + 1] + rho_original[i + 1]);
            }
        }
        apply_map("cp_map", cp_grid);
        cp_max = std::max(cp, *std::max_element(cp_grid.begin(), cp_grid.end()));
        const float courant = cp_max * dt * std::sqrt(1.0f/(roi.dx*roi.dx) + 1.0f/(roi.dz*roi.dz));
        std::cout << "\nNumero de Courant e " << courant << '\n';
        if (courant > 1.0f) throw std::runtime_error("O passo de tempo e muito longo; Courant=" + std::to_string(courant));
    }

    void build_probes(const Json::Value& root) {
        const Json::Value probe_cfg = root["probes"];
        if (!probe_cfg.isArray() || probe_cfg.empty()) throw std::invalid_argument("A configuracao precisa conter ao menos um probe");
        for (const auto& node : probe_cfg) probes.push_back(parse_probe(node, roi));
        const float alpha_max = kPi * probes.front().freq;
        (void)alpha_max;

        source_map.assign(static_cast<std::size_t>(roi.nx()) * roi.ny(), -1);
        sensor_map.assign(source_map.size(), -1);
        for (int pi = 0; pi < static_cast<int>(probes.size()); ++pi) {
            for (int ei = 0; ei < static_cast<int>(probes[pi].elements.size()); ++ei) {
                auto& e = probes[pi].elements[ei];
                if (e.tx && !e.cells.empty()) {
                    e.tx_channel = static_cast<int>(tx_channels.size());
                    tx_channels.emplace_back(pi, ei);
                    for (int cell : e.cells) source_map[cell] = e.tx_channel;
                }
                if (e.rx && !e.cells.empty()) {
                    e.rx_channel = static_cast<int>(rx_channels.size());
                    rx_channels.emplace_back(pi, ei);
                    for (int cell : e.cells) sensor_map[cell] = e.rx_channel;
                    receiver_delays.push_back(static_cast<int>(e.rx_delay / dt + 1.0f));
                }
            }
        }
        if (tx_channels.empty()) throw std::invalid_argument("Nenhum emissor ativo localizado na ROI");
        if (rx_channels.empty()) throw std::invalid_argument("Nenhum receptor ativo localizado na ROI");
    }

    void build_pml() {
        const float alpha_max = kPi * probes.front().freq;
        pml[0] = roi.pml('x', false, dt, cp_max, alpha_max);
        pml[1] = roi.pml('x', true, dt, cp_max, alpha_max);
        pml[2] = roi.pml('x', false, dt, cp_max, alpha_max);
        pml[3] = roi.pml('z', false, dt, cp_max, alpha_max);
        pml[4] = roi.pml('z', true, dt, cp_max, alpha_max);
        pml[5] = roi.pml('z', false, dt, cp_max, alpha_max);
    }

    void read_run_options(const Json::Value& root) {
        const Json::Value opts = root.get("simul_configs", Json::Value(Json::objectValue));
        n_iter = integer(opts, "n_iter", 1);
        if (n_iter <= 0) throw std::invalid_argument("n_iter precisa ser positivo");
        show_anim = flag(opts, "show_anim");
        show_debug = flag(opts, "show_debug");
        show_figs = flag(opts, "show_figs");
        plot_results = flag(opts, "plot_results");
        plot_error = flag(opts, "plot_error");
        plot_sensors = flag(opts, "plot_sensors");
        plot_bscan = flag(opts, "plot_bscan");
        save_results = flag(opts, "save_results");
        save_field = flag(opts, "save_field");
        save_sensors = flag(opts, "save_sensors");
        save_bscan = flag(opts, "save_bscan");
        save_sources = flag(opts, "save_sources");
        source_env = flag(opts, "source_env");
        min_fields = number(opts, "min_val_fields", -100.0f);
        max_fields = number(opts, "max_val_fields", 100.0f);
        const fs::path requested = text(opts, "results_dir", ".");
        if (fs::is_directory(requested)) results_dir = requested;
        else results_dir = ".";
    }

    void build_default_sources() { source_term = make_sources(nullptr); }

    void load_optional_law(const Json::Value& root) {
        const Json::Value opts = root.get("simul_configs", Json::Value(Json::objectValue));
        if (!opts.isMember("emission_laws")) return;
        const fs::path path = text(opts, "emission_laws");
        if (fs::is_regular_file(path)) laws = load_law(path);
    }

    std::vector<float> make_sources(const std::vector<float>* delays) const {
        const std::size_t nsrc = tx_channels.size();
        std::vector<float> result(static_cast<std::size_t>(n_steps) * nsrc, 0.0f);
        const int ord_source = sim_model == "unsplit" ? 2 : 1;
        for (std::size_t channel = 0; channel < nsrc; ++channel) {
            const auto [pi, ei] = tx_channels[channel];
            Element element = probes[pi].elements[ei];
            if (delays && ei < static_cast<int>(delays->size())) element.tx_delay = (*delays)[ei];
            const auto waveform = make_source(element, n_steps, dt, ord_source, source_env, probes[pi].linear);
            for (int t = 0; t < n_steps; ++t) result[static_cast<std::size_t>(t) * nsrc + channel] = waveform[t];
        }
        return result;
    }

    sim_cuda_cpp::SimulationInput input_for(const std::vector<float>& source, std::function<void(const std::vector<float>&, int, float)> callback) const {
        sim_cuda_cpp::SimulationInput in;
        in.nx = roi.nx(); in.ny = roi.ny(); in.n_steps = n_steps;
        in.n_sources = static_cast<int>(tx_channels.size()); in.n_receivers = static_cast<int>(rx_channels.size());
        in.ord = static_cast<int>(coefficients.size()); in.it_display = it_display;
        in.pml_x_size = roi.nx() - 2 * (in.ord - 1);
        in.pml_y_size = roi.ny() - 2 * (in.ord - 1);
        in.dt = dt; in.dx = roi.dx; in.dy = roi.dz;
        in.coefficients = coefficients.data();
        in.rho_grid_vx = rho_grid.data(); in.rho_grid_vy = rho_grid.data(); in.cp_grid_vx = cp_grid.data();
        in.a_x = pml[0].a.data(); in.b_x = pml[0].b.data(); in.k_x = pml[0].k.data();
        in.a_x_half = pml[1].a.data(); in.b_x_half = pml[1].b.data(); in.k_x_half = pml[1].k.data();
        in.a_y = pml[3].a.data(); in.b_y = pml[3].b.data(); in.k_y = pml[3].k.data();
        in.a_y_half = pml[4].a.data(); in.b_y_half = pml[4].b.data(); in.k_y_half = pml[4].k.data();
        in.source_map = source_map.data(); in.source_term = source.data();
        in.sensor_map = sensor_map.data(); in.receiver_delay = receiver_delays.data();
        in.show_debug = show_debug; in.frame_callback = std::move(callback);
        return in;
    }
};

cv::Mat field_image(const std::vector<float>& field, const Roi& roi, float min_value, float max_value) {
    cv::Mat image(roi.iy_max() - roi.iy_min(), roi.ix_max() - roi.ix_min(), CV_8UC1);
    const float span = std::max(max_value - min_value, 1e-20f);
    for (int y = roi.iy_min(); y < roi.iy_max(); ++y) {
        auto* row = image.ptr<unsigned char>(y - roi.iy_min());
        for (int x = roi.ix_min(); x < roi.ix_max(); ++x) {
            const float v = field[static_cast<std::size_t>(x) * roi.ny() + y];
            row[x - roi.ix_min()] = cv::saturate_cast<unsigned char>((v - min_value) * (255.0f / span));
        }
    }
    return image;
}

cv::Mat error_image(const std::vector<float>& pressure, const NpyArray& reference, const Roi& roi) {
    cv::Mat values(roi.iy_max() - roi.iy_min(), roi.ix_max() - roi.ix_min(), CV_32FC1);
    for (int y = roi.iy_min(); y < roi.iy_max(); ++y) for (int x = roi.ix_min(); x < roi.ix_max(); ++x) {
        const auto ri = static_cast<std::size_t>(x - roi.ix_min()) * reference.shape[1] + (y - roi.iy_min());
        values.at<float>(y - roi.iy_min(), x - roi.ix_min()) = reference.values[ri] - pressure[static_cast<std::size_t>(x) * roi.ny() + y];
    }
    cv::Mat normalized, color;
    cv::normalize(values, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
    return color;
}

cv::Mat bscan_image(const std::vector<float>& values, int n_steps, int n_receivers) {
    if (n_receivers <= 0 || n_steps <= 0) return {};
    cv::Mat data(n_steps, n_receivers, CV_32FC1, const_cast<float*>(values.data()));
    cv::Mat normalized, color;
    cv::normalize(data, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(normalized, color, cv::COLORMAP_VIRIDIS);
    return color;
}

cv::Mat signal_plot(const std::vector<float>& signal, const std::vector<float>* reference = nullptr,
                    std::optional<float> echo_sample = std::nullopt) {
    const int width = 1000, height = 400;
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::line(image, {40, height - 30}, {width - 10, height - 30}, cv::Scalar(0, 0, 0), 1);
    cv::line(image, {40, 10}, {40, height - 30}, cv::Scalar(0, 0, 0), 1);
    float max_abs = 1e-20f;
    for (float v : signal) max_abs = std::max(max_abs, std::fabs(v));
    if (reference) for (float v : *reference) max_abs = std::max(max_abs, std::fabs(v));
    auto draw = [&](const std::vector<float>& values, cv::Scalar color) {
        if (values.size() < 2) return;
        cv::Point previous;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const int x = 40 + static_cast<int>((width - 55) * i / (values.size() - 1));
            const int y = height / 2 - static_cast<int>((height / 2 - 25) * values[i] / max_abs);
            cv::Point current(x, y);
            if (i) cv::line(image, previous, current, color, 1, cv::LINE_AA);
            previous = current;
        }
    };
    draw(signal, cv::Scalar(220, 70, 20));
    if (reference) draw(*reference, cv::Scalar(20, 100, 220));
    if (echo_sample && signal.size() > 1 && *echo_sample >= 0.0f && *echo_sample < signal.size()) {
        const int x = 40 + static_cast<int>((width - 55) * *echo_sample / (signal.size() - 1));
        cv::line(image, {x, 10}, {x, height - 30}, cv::Scalar(20, 180, 20), 1, cv::LINE_AA);
    }
    return image;
}

std::optional<float> expected_echo_sample(const Prepared& sim) {
    std::optional<std::array<float, 3>> emitter, receiver;
    float emission_delay = 0.0f;
    for (const auto& probe : sim.probes) {
        const bool all_tx = !probe.elements.empty() && std::all_of(probe.elements.begin(), probe.elements.end(),
            [](const Element& e) { return e.tx; });
        const bool all_rx = !probe.elements.empty() && std::all_of(probe.elements.begin(), probe.elements.end(),
            [](const Element& e) { return e.rx; });
        if (!emitter && all_tx) {
            emitter = probe.center;
            emission_delay = probe.elements.front().tx_delay;
        }
        if (!receiver && all_rx) receiver = probe.center;
    }
    if (!emitter || !receiver) return std::nullopt;
    float distance2 = 0.0f;
    for (int axis = 0; axis < 3; ++axis) distance2 += std::pow((*emitter)[axis] - (*receiver)[axis], 2.0f);
    return (std::sqrt(distance2) / sim.cp + emission_delay) / sim.dt;
}

struct RunRow { double gpu = 0, total = 0, mse_pressure = std::numeric_limits<double>::quiet_NaN(), mse_sensor = std::numeric_limits<double>::quiet_NaN(); };

std::optional<double> mse(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return std::nullopt;
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        sum += d * d;
    }
    return sum / a.size();
}

std::string timestamp() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&now, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return out.str();
}

std::vector<float> cropped_pressure(const std::vector<float>& pressure, const Roi& roi) {
    std::vector<float> cropped;
    cropped.reserve(static_cast<std::size_t>(roi.ix_max() - roi.ix_min()) * (roi.iy_max() - roi.iy_min()));
    for (int x = roi.ix_min(); x < roi.ix_max(); ++x) for (int y = roi.iy_min(); y < roi.iy_max(); ++y) {
        cropped.push_back(pressure[static_cast<std::size_t>(x) * roi.ny() + y]);
    }
    return cropped;
}

std::optional<NpyArray> crop_pressure_reference(const NpyArray& reference, const Roi& roi) {
    if (reference.shape.size() != 2) return std::nullopt;
    const std::size_t roi_nx = static_cast<std::size_t>(roi.ix_max() - roi.ix_min());
    const std::size_t roi_ny = static_cast<std::size_t>(roi.iy_max() - roi.iy_min());
    NpyArray cropped;
    if (reference.shape[0] == static_cast<std::size_t>(roi.nx()) &&
        reference.shape[1] == static_cast<std::size_t>(roi.ny())) {
        // Python carrega o campo de referência completo e aplica os mesmos slices da ROI.
        cropped.shape = {roi_nx, roi_ny};
        cropped.values.reserve(roi_nx * roi_ny);
        for (int x = roi.ix_min(); x < roi.ix_max(); ++x) {
            for (int y = roi.iy_min(); y < roi.iy_max(); ++y) {
                cropped.values.push_back(reference.values[static_cast<std::size_t>(x) * roi.ny() + y]);
            }
        }
        return cropped;
    }
    // Também aceita referências já recortadas, desde que o conteúdo tenha o tamanho esperado.
    if (reference.shape[0] == roi_nx && reference.shape[1] == roi_ny &&
        reference.values.size() == roi_nx * roi_ny) {
        return reference;
    }
    return std::nullopt;
}

std::vector<float> stats_values(const std::vector<double>& values, int start = 0) {
    std::vector<float> out;
    for (std::size_t i = static_cast<std::size_t>(std::max(start, 0)); i < values.size(); ++i) out.push_back(static_cast<float>(values[i]));
    return out;
}

double mean(const std::vector<double>& values, std::size_t start) {
    if (start >= values.size()) return std::numeric_limits<double>::quiet_NaN();
    return std::accumulate(values.begin() + start, values.end(), 0.0) / (values.size() - start);
}

double stddev(const std::vector<double>& values, std::size_t start) {
    if (start >= values.size()) return std::numeric_limits<double>::quiet_NaN();
    const double avg = mean(values, start);
    double sum = 0;
    for (std::size_t i = start; i < values.size(); ++i) sum += std::pow(values[i] - avg, 2);
    return std::sqrt(sum / (values.size() - start));
}

int run_simulation(Prepared& sim) {
    const std::string started = timestamp();
    std::cout << "Simulacao " << sim.name << '\n';
    const int n_laws = sim.laws ? static_cast<int>(sim.laws->delay.size()) : 1;
    const auto echo_sample = expected_echo_sample(sim);
    std::vector<double> gpu_times, total_times, mse_pressure_values, mse_sensor_values;
    std::vector<float> last_sources = sim.source_term;
    std::string last_gpu;
    int last_iter = 0, last_law = 0;
    const int n_receivers = static_cast<int>(sim.rx_channels.size());

    for (int iteration = 0; iteration < sim.n_iter; ++iteration) {
        std::cout << "Iteracao " << iteration << '\n';
        for (int law = 0; law < n_laws; ++law) {
            std::cout << "\tLaw " << law << " of " << n_laws << '\n';
            const std::vector<float>* delays = sim.laws ? &sim.laws->delay[law] : nullptr;
            last_sources = sim.make_sources(delays);
            std::function<void(const std::vector<float>&, int, float)> callback;
            if (sim.show_anim) callback = [&](const std::vector<float>& frame, int it, float max_pressure) {
                (void)it; (void)max_pressure;
                cv::imshow("Pressure", field_image(frame, sim.roi, sim.min_fields, sim.max_fields));
                cv::waitKey(1);
            };
            const auto total_start = std::chrono::steady_clock::now();
            const auto result = sim_cuda_cpp::run(sim.input_for(last_sources, std::move(callback)));
            const auto total_end = std::chrono::steady_clock::now();
            const double total_seconds = std::chrono::duration<double>(total_end - total_start).count();
            gpu_times.push_back(result.sim_time_seconds);
            total_times.push_back(total_seconds);
            last_gpu = result.gpu_name;
            last_iter = iteration;
            last_law = law;
            std::cout << result.gpu_name << '\n'
                      << std::fixed << std::setprecision(3) << result.sim_time_seconds << "s\n"
                      << "Tempo total (inclui preparacao/transferencia): " << total_seconds << "s\n";

            const fs::path pressure_ref_path = sim.results_dir / ("result_ref_" + sim.sim_model + "_field_pressure.npy");
            const fs::path sensor_ref_path = sim.results_dir / ("result_ref_" + sim.sim_model + "_bscan_pressure.npy");
            if (fs::is_regular_file(pressure_ref_path)) {
                const auto ref = load_npy(pressure_ref_path);
                const auto cropped_ref = crop_pressure_reference(ref, sim.roi);
                const auto current = cropped_pressure(result.pressure, sim.roi);
                std::cout << "ref.shape: ";
                if (ref.shape.size() == 2) std::cout << ref.shape[0] << "x" << ref.shape[1];
                else std::cout << "rank " << ref.shape.size();
                std::cout << ", current ROI.shape: " << (sim.roi.ix_max() - sim.roi.ix_min()) << "x"
                          << (sim.roi.iy_max() - sim.roi.iy_min()) << '\n';
                if (cropped_ref) {
                    const auto value = mse(cropped_ref->values, current);
                    if (value) {
                        mse_pressure_values.push_back(*value);
                        std::cout << "MSE do campo de pressao: " << std::scientific << std::setprecision(4)
                                  << *value << std::defaultfloat << '\n';
                    }
                    if (sim.plot_error) {
                        cv::Mat im = error_image(result.pressure, *cropped_ref, sim.roi);
                        cv::imshow("Pressure error", im);
                    }
                } else {
                    mse_pressure_values.push_back(std::numeric_limits<double>::infinity());
                    std::cout << "MSE do campo de pressao: inf (referencia nao corresponde a grade completa nem a ROI)\n";
                }
            } else {
                std::cout << "Referencia de pressao nao encontrada: " << pressure_ref_path << '\n';
            }

            std::optional<NpyArray> sensor_ref;
            if (fs::is_regular_file(sensor_ref_path)) {
                sensor_ref = load_npy(sensor_ref_path);
                const bool sensor_shape_matches = sensor_ref->shape.size() == 2 &&
                    sensor_ref->shape[0] == static_cast<std::size_t>(sim.n_steps) &&
                    sensor_ref->shape[1] == static_cast<std::size_t>(n_receivers) &&
                    sensor_ref->values.size() == result.sensor_pressure.size();
                std::cout << "sens_ref.shape: ";
                if (sensor_ref->shape.size() == 2) std::cout << sensor_ref->shape[0] << "x" << sensor_ref->shape[1];
                else std::cout << "rank " << sensor_ref->shape.size();
                std::cout << ", sens_pressure.shape: " << sim.n_steps << "x" << n_receivers << '\n';
                if (sensor_shape_matches) {
                    const auto value = mse(sensor_ref->values, result.sensor_pressure);
                    if (value) {
                        mse_sensor_values.push_back(*value);
                        std::cout << "MSE dos sensores de pressao: " << std::scientific << std::setprecision(4)
                                  << *value << std::defaultfloat << '\n';
                    }
                } else {
                    mse_sensor_values.push_back(std::numeric_limits<double>::infinity());
                    std::cout << "MSE dos sensores de pressao: inf (referencia e resultado devem ter shape "
                              << sim.n_steps << "x" << n_receivers << ")\n";
                }
            } else {
                std::cout << "Referencia dos sensores nao encontrada: " << sensor_ref_path << '\n';
            }
            std::optional<NpyArray> sensor_plot_ref;
            const fs::path legacy_plot_ref = "ensaios/ponto/results/result_ref_unsplit_bscan_pressure.npy";
            if (sim.plot_sensors && fs::is_regular_file(legacy_plot_ref)) sensor_plot_ref = load_npy(legacy_plot_ref);

            const std::string base = (sim.results_dir / ("result_" + sim.name + "_" + started + "_" +
                std::to_string(sim.roi.nx()) + "x" + std::to_string(sim.roi.ny()) + "_" + std::to_string(sim.n_steps) +
                "_iter_" + std::to_string(iteration) + "_law_" + std::to_string(law))).string();
            if (sim.plot_results) {
                cv::Mat image = field_image(result.pressure, sim.roi, sim.min_fields, sim.max_fields);
                cv::imshow("Pressure result", image);
                if (sim.save_results) cv::imwrite(base + "_field_pressure.png", image);
            }
            if (sim.plot_error && !pressure_ref_path.empty() && !fs::is_regular_file(pressure_ref_path)) {
                std::cerr << "Plot de erro ignorado: referencia nao encontrada\n";
            }
            if (sim.plot_sensors) {
                for (int receiver = 0; receiver < n_receivers; ++receiver) {
                    std::vector<float> signal(sim.n_steps);
                    for (int it = 0; it < sim.n_steps; ++it) signal[it] = result.sensor_pressure[static_cast<std::size_t>(it) * n_receivers + receiver];
                    std::optional<std::vector<float>> ref_signal;
                    if (sensor_plot_ref && sensor_plot_ref->shape.size() == 1) {
                        ref_signal = sensor_plot_ref->values;
                    } else if (sensor_plot_ref && sensor_plot_ref->shape.size() == 2 &&
                               sensor_plot_ref->shape[0] == static_cast<std::size_t>(sim.n_steps) &&
                               receiver < static_cast<int>(sensor_plot_ref->shape[1])) {
                        ref_signal.emplace(sim.n_steps);
                        for (int it = 0; it < sim.n_steps; ++it) {
                            (*ref_signal)[it] = sensor_plot_ref->values[static_cast<std::size_t>(it) * sensor_plot_ref->shape[1] + receiver];
                        }
                    }
                    cv::Mat image = signal_plot(signal, ref_signal ? &*ref_signal : nullptr, echo_sample);
                    cv::imshow("Receiver " + std::to_string(receiver + 1), image);
                    if (sim.save_sensors) cv::imwrite(base + "_sensor_" + std::to_string(receiver) + ".png", image);
                }
            }
            if (sim.plot_bscan) {
                cv::Mat image = bscan_image(result.sensor_pressure, sim.n_steps, n_receivers);
                if (!image.empty()) {
                    cv::imshow("B-scan pressure", image);
                    if (sim.save_results) cv::imwrite(base + "_bscan_pressure.png", image);
                }
            }
            if (sim.save_results && sim.save_field) {
                save_npy(base + "_field_pressure.npy", result.pressure, {static_cast<std::size_t>(sim.roi.nx()), static_cast<std::size_t>(sim.roi.ny())});
            }
            if (sim.save_bscan) {
                save_npy(base + "_bscan_pressure.npy", result.sensor_pressure,
                         {static_cast<std::size_t>(sim.n_steps), static_cast<std::size_t>(n_receivers)});
            }
            if (sim.show_figs) cv::waitKey(1);
        }
    }

    std::cout << "TEMPO - " << sim.n_steps << " pontos de tempo\n";
    if (sim.n_iter > 5 && gpu_times.size() > 5) {
        std::cout << "Tempo medio de execucao: " << mean(gpu_times, 5) << "s (std = " << stddev(gpu_times, 5) << ")\n"
                  << "Tempo medio total: " << mean(total_times, 5) << "s (std = " << stddev(total_times, 5) << ")\n";
    }
    if (sim.n_iter > 5 && mse_pressure_values.size() > 5) {
        std::cout << "MSE medio do campo de pressao: " << mean(mse_pressure_values, 5) << '\n';
    }
    if (sim.n_iter > 5 && mse_sensor_values.size() > 5) {
        std::cout << "MSE medio dos sensores: " << mean(mse_sensor_values, 5) << '\n';
    }

    if (sim.save_sources) {
        save_npy((sim.results_dir / ("sources_" + sim.name + "_" + started + ".npy")), last_sources,
                 {static_cast<std::size_t>(sim.n_steps), sim.tx_channels.size()});
    }
    if (sim.save_results && !gpu_times.empty()) {
        const std::string base = (sim.results_dir / ("result_" + sim.name + "_" + started + "_" +
            std::to_string(sim.roi.nx()) + "x" + std::to_string(sim.roi.ny()) + "_" + std::to_string(sim.n_steps) +
            "_iter_" + std::to_string(last_iter) + "_")).string();
        std::ofstream csv(base + "GPU_.csv");
        for (double value : gpu_times) csv << std::fixed << std::setprecision(3) << value << '\n';
        std::ofstream desc(base + "_desc.txt");
        desc << "Parametros do ensaio\n--------------------\n\n"
             << "Simulador: " << sim.name << '\n' << "Modelo do simulador: " << sim.sim_model << '\n'
             << "Quantidade de iteracoes no tempo: " << sim.n_steps << '\n'
             << "Tamanho da ROI: " << sim.roi.nx() << 'x' << sim.roi.ny() << '\n'
             << "GPU: " << last_gpu << '\n' << "Numero de simulacoes: " << sim.n_iter << '\n'
             << "Block size: (16, 16)\n";
        if (sim.n_iter > 5 && gpu_times.size() > 5) {
            desc << "Tempo medio de execucao: " << mean(gpu_times, 5) << "s\nDesvio padrao: " << stddev(gpu_times, 5) << '\n'
                 << "Tempo medio total: " << mean(total_times, 5) << "s\nDesvio padrao: " << stddev(total_times, 5) << '\n';
            if (mse_pressure_values.size() > 5) desc << "MSE medio do campo de pressao: " << mean(mse_pressure_values, 5) << '\n';
            if (mse_sensor_values.size() > 5) desc << "MSE medio dos sensores de pressao: " << mean(mse_sensor_values, 5) << '\n';
        } else {
            desc << "Tempo execucao: " << gpu_times.back() << "s\nTempo total: " << total_times.back() << "s\n";
            if (!mse_pressure_values.empty()) desc << "MSE do campo de pressao: " << mse_pressure_values.back() << '\n';
            if (!mse_sensor_values.empty()) desc << "MSE dos sensores de pressao: " << mse_sensor_values.back() << '\n';
        }
    }
    if (sim.show_figs) cv::waitKey(1);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string config_path = "config.json";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
                config_path = argv[++i];
            } else if (arg.rfind("--config=", 0) == 0) {
                config_path = arg.substr(9);
            } else if (arg.rfind("-c=", 0) == 0) {
                config_path = arg.substr(3);
            }
            else if (arg == "-h" || arg == "--help") {
                std::cout << "Uso: simulator_cuda_cpp [-c|--config arquivo.json|--config=arquivo.json]\n";
                return 0;
            } else {
                throw std::invalid_argument("Argumento desconhecido: " + arg);
            }
        }
        Json::Value config;
        Json::CharReaderBuilder builder;
        std::string errors;
        std::ifstream file(config_path);
        if (!file) throw std::runtime_error("Arquivo de configuracao nao encontrado: " + config_path);
        if (!Json::parseFromStream(builder, file, &config, &errors)) {
            throw std::runtime_error("Falha ao ler configuracao: " + errors);
        }
        Prepared simulator(config, "split");
        return run_simulation(simulator);
    } catch (const std::exception& error) {
        std::cerr << "Erro: " << error.what() << '\n';
        return 1;
    }
}
