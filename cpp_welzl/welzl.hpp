#ifndef WELZL_HPP
#define WELZL_HPP

#include <vector>
#include <functional>

// Helper structure representing a bounding ball.
struct Ball {
    std::vector<double> center;
    double radius_sq = -1.0;
    bool success = false;
};

// C interface for Python ctypes integration.
extern "C" {
    bool welzl_consent(
        const double* points,
        const int* original_indices,
        int num_points,
        int dim,
        bool (*consent_callback)(int),
        double* out_center,
        double* out_radius_sq
    );
}

// Internal functions (exposed for testing)
Ball get_circum_ball(const std::vector<std::vector<double>>& R, int dim);
bool solve_linear_system(const std::vector<std::vector<double>>& M, const std::vector<double>& B, std::vector<double>& x);

Ball welzl_impl(
    const std::vector<std::vector<double>>& P,
    std::vector<std::vector<double>> R,
    int dim,
    int n,
    const std::function<bool(int)>& consent_callback,
    const std::vector<int>& p_indices
);

#endif // WELZL_HPP
