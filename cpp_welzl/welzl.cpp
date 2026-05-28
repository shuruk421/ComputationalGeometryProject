#include "welzl.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

// Solves M * x = B using Gaussian elimination with partial pivoting.
bool solve_linear_system(const std::vector<std::vector<double>>& M, const std::vector<double>& B, std::vector<double>& x) {
    int n = M.size();
    if (n == 0) return true;
    
    // Create augmented matrix
    std::vector<std::vector<double>> A(n, std::vector<double>(n + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = M[i][j];
        }
        A[i][n] = B[i];
    }
    
    // Gaussian elimination with partial pivoting
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int row = i + 1; row < n; ++row) {
            if (std::abs(A[row][i]) > std::abs(A[pivot][i])) {
                pivot = row;
            }
        }
        
        if (pivot != i) {
            std::swap(A[i], A[pivot]);
        }
        
        if (std::abs(A[i][i]) < 1e-11) {
            return false;
        }
        
        for (int row = i + 1; row < n; ++row) {
            double factor = A[row][i] / A[i][i];
            for (int col = i; col <= n; ++col) {
                A[row][col] -= factor * A[i][col];
            }
        }
    }
    
    // Back substitution
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (A[i][n] - sum) / A[i][i];
    }
    
    return true;
}

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    int n = a.size();
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

Ball get_circum_ball(const std::vector<std::vector<double>>& R, int dim) {
    Ball result;
    if (R.empty()) {
        result.success = false;
        return result;
    }
    
    if (R.size() == 1) {
        result.center = R[0];
        result.radius_sq = 0.0;
        result.success = true;
        return result;
    }
    
    int n = R.size() - 1;
    std::vector<std::vector<double>> vs(n, std::vector<double>(dim));
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            vs[i][d] = R[i + 1][d] - R[0][d];
        }
    }
    
    std::vector<std::vector<double>> M(n, std::vector<double>(n));
    std::vector<double> B(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            M[i][j] = 2.0 * dot_product(vs[i], vs[j]);
        }
        B[i] = dot_product(vs[i], vs[i]);
    }
    
    std::vector<double> lambdas;
    if (!solve_linear_system(M, B, lambdas)) {
        result.success = false;
        return result;
    }
    
    result.center.assign(dim, 0.0);
    for (int d = 0; d < dim; ++d) {
        double delta_c_d = 0.0;
        for (int j = 0; j < n; ++j) {
            delta_c_d += lambdas[j] * vs[j][d];
        }
        result.center[d] = R[0][d] + delta_c_d;
    }
    
    double delta_c_sq = 0.0;
    for (int d = 0; d < dim; ++d) {
        double delta_c_d = result.center[d] - R[0][d];
        delta_c_sq += delta_c_d * delta_c_d;
    }
    result.radius_sq = delta_c_sq;
    result.success = true;
    
    return result;
}

// Forward declaration of recursive helper
Ball welzl_impl(
    const std::vector<std::vector<double>>& P,
    std::vector<std::vector<double>> R,
    int dim,
    int n,
    const std::function<bool(int)>& consent_callback,
    const std::vector<int>& p_indices
);

// Circumsphere solver with fallback for degenerate configurations
Ball get_circum_ball_with_fallback(
    const std::vector<std::vector<double>>& R,
    int dim
) {
    if (R.empty()) {
        Ball b;
        b.success = true;
        b.radius_sq = -1.0;
        return b;
    }
    Ball b = get_circum_ball(R, dim);
    if (!b.success) {
        // Fallback for degenerate boundary points: run Welzl on R itself
        std::vector<int> r_indices(R.size(), -1);
        auto dummy_callback = [](int) -> bool { return true; };
        b = welzl_impl(R, {}, dim, static_cast<int>(R.size()), dummy_callback, r_indices);
    }
    return b;
}

// Recursive implementation of Welzl's Minimum Enclosing Ball algorithm with consent checks.
// p_indices maps the shuffled points in P to their original pre-shuffled indices.
// This ensures that consent_callback is queried with the correct original point index.
Ball welzl_impl(
    const std::vector<std::vector<double>>& P,
    std::vector<std::vector<double>> R,
    int dim,
    int n,
    const std::function<bool(int)>& consent_callback,
    const std::vector<int>& p_indices
) {
    if (n == 0 || static_cast<int>(R.size()) == dim + 1) {
        return get_circum_ball_with_fallback(R, dim);
    }
    
    Ball b = get_circum_ball_with_fallback(R, dim);
    
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
        
        // orig_idx is the original pre-shuffled index of the point.
        // A value >= 0 indicates a point from the original dataset requiring a consent check.
        // A value < 0 (e.g. in fallback runs on degenerate base cases) skips the consent check.
        int orig_idx = p_indices[i];
        bool consented = true;
        if (orig_idx >= 0) {
            consented = consent_callback(orig_idx);
        }
        
        if (consented) {
            auto next_R = R;
            next_R.push_back(pt);
            Ball next_b = welzl_impl(P, next_R, dim, i, consent_callback, p_indices);
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

extern "C" {
    bool welzl_consent(
        const double* points,
        const int* original_indices,
        int num_points,
        int dim,
        bool (*consent_callback)(int),
        double* out_center,
        double* out_radius_sq
    ) {
        if (num_points == 0) {
            return false;
        }
        
        std::vector<std::vector<double>> P(num_points, std::vector<double>(dim));
        for (int i = 0; i < num_points; ++i) {
            for (int d = 0; d < dim; ++d) {
                P[i][d] = points[i * dim + d];
            }
        }
        
        std::vector<int> p_indices(original_indices, original_indices + num_points);
        
        Ball b = welzl_impl(P, {}, dim, num_points, consent_callback, p_indices);
        
        if (b.success && b.radius_sq >= 0.0) {
            for (int d = 0; d < dim; ++d) {
                out_center[d] = b.center[d];
            }
            *out_radius_sq = b.radius_sq;
            return true;
        }
        
        return false;
    }
}
