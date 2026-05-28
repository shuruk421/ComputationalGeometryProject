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
    
    auto callback = [&](int orig_idx) -> bool {
        if (orig_idx >= 0) {
            if (!queried[orig_idx]) {
                queried[orig_idx] = true;
                res.oracle_calls++;
            }
            return consents[orig_idx];
        }
        return true;
    };
    
    Ball b = welzl_impl(P, {}, dim, static_cast<int>(P.size()), callback, p_indices);
    
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
