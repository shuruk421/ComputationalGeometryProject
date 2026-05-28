#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "welzl.hpp"

namespace py = pybind11;

// Result struct to return to Python
struct WelzlResult {
    std::vector<double> center;
    double radius_sq;
    int oracle_calls;
    bool success;
};

// Internal hybrid recursive-loop solver
Ball welzl_impl_cpp(
    const std::vector<std::vector<double>>& P,
    std::vector<std::vector<double>> R,
    int dim,
    int n,
    const std::vector<bool>& consents,
    const std::vector<int>& p_indices,
    int& oracle_calls,
    std::vector<bool>& queried
) {
    if (n == 0 || static_cast<int>(R.size()) == dim + 1) {
        if (R.empty()) {
            Ball b;
            b.success = true;
            b.radius_sq = -1.0;
            return b;
        }
        Ball b = get_circum_ball(R, dim);
        if (!b.success) {
            std::vector<int> r_indices(R.size(), -1);
            b = welzl_impl_cpp(R, {}, dim, static_cast<int>(R.size()), consents, r_indices, oracle_calls, queried);
        }
        return b;
    }
    
    Ball b = get_circum_ball(R, dim);
    if (!b.success && !R.empty()) {
        std::vector<int> r_indices(R.size(), -1);
        b = welzl_impl_cpp(R, {}, dim, static_cast<int>(R.size()), consents, r_indices, oracle_calls, queried);
    }
    
    double radius_sq = b.radius_sq;
    std::vector<double> center = b.center;
    bool has_ball = b.success && !center.empty();
    
    for (int i = 0; i < n; ++i) {
        const auto& pt = P[i];
        
        bool inside = false;
        if (has_ball) {
            double dist_sq = 0.0;
            for (int d = 0; d < dim; ++d) {
                double diff = pt[d] - center[d];
                dist_sq += diff * diff;
            }
            if (dist_sq <= radius_sq + 1e-10) {
                inside = true;
            }
        }
        
        if (inside) continue;
        
        int orig_idx = p_indices[i];
        bool consented = true;
        if (orig_idx >= 0) {
            if (!queried[orig_idx]) {
                queried[orig_idx] = true;
                oracle_calls++;
            }
            consented = consents[orig_idx];
        }
        
        if (consented) {
            auto next_R = R;
            next_R.push_back(pt);
            Ball next_b = welzl_impl_cpp(P, next_R, dim, i, consents, p_indices, oracle_calls, queried);
            if (next_b.success) {
                b = next_b;
                radius_sq = b.radius_sq;
                center = b.center;
                has_ball = true;
            }
        }
    }
    
    return b;
}

WelzlResult welzl_consent_cpp(
    const std::vector<std::vector<double>>& P,
    const std::vector<bool>& consents,
    int dim
) {
    WelzlResult res;
    res.success = false;
    res.radius_sq = -1.0;
    res.oracle_calls = 0;
    
    if (P.empty()) {
        return res;
    }
    
    std::vector<int> p_indices(P.size());
    for (size_t i = 0; i < P.size(); ++i) {
        p_indices[i] = i;
    }
    
    // Direct-mapped cache for unique consent queries
    std::vector<bool> queried(P.size(), false);
    
    Ball b = welzl_impl_cpp(P, {}, dim, static_cast<int>(P.size()), consents, p_indices, res.oracle_calls, queried);
    
    if (b.success && b.radius_sq >= 0.0) {
        res.center = b.center;
        res.radius_sq = b.radius_sq;
        res.success = true;
    }
    
    return res;
}

PYBIND11_MODULE(welzl_cpp_module, m) {
    m.doc() = "C++ Welzl Minimum Enclosing Ball solver with pybind11";
    
    py::class_<WelzlResult>(m, "WelzlResult")
        .def_readonly("center", &WelzlResult::center)
        .def_readonly("radius_sq", &WelzlResult::radius_sq)
        .def_readonly("oracle_calls", &WelzlResult::oracle_calls)
        .def_readonly("success", &WelzlResult::success);
        
    m.def("welzl_consent_cpp", &welzl_consent_cpp,
          "Computes the Minimum Enclosing Ball using consent tracking in C++",
          py::arg("points"), py::arg("consents"), py::arg("dim"));
}
