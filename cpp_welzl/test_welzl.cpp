#include "welzl.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

void test_solve_linear_system() {
    std::cout << "Testing solve_linear_system..." << std::endl;
    
    // Test 2x2 system
    std::vector<std::vector<double>> M = {{2.0, 1.0}, {1.0, 3.0}};
    std::vector<double> B = {5.0, 5.0};
    std::vector<double> x;
    bool success = solve_linear_system(M, B, x);
    assert(success);
    assert(std::abs(x[0] - 2.0) < 1e-9);
    assert(std::abs(x[1] - 1.0) < 1e-9);
    
    // Test singular system
    std::vector<std::vector<double>> M_singular = {{1.0, 2.0}, {2.0, 4.0}};
    std::vector<double> B_singular = {3.0, 6.0};
    success = solve_linear_system(M_singular, B_singular, x);
    assert(!success);
    
    std::cout << "solve_linear_system passed!" << std::endl;
}

void test_get_circum_ball() {
    std::cout << "Testing get_circum_ball..." << std::endl;
    
    // Test circumsphere of a 2D triangle: (0,0), (2,0), (1,1)
    // The circumcenter is (1.0, 0.0), and radius_sq is 1.0
    std::vector<std::vector<double>> R = {{0.0, 0.0}, {2.0, 0.0}, {1.0, 1.0}};
    Ball b = get_circum_ball(R, 2);
    assert(b.success);
    assert(std::abs(b.center[0] - 1.0) < 1e-9);
    assert(std::abs(b.center[1] - 0.0) < 1e-9);
    assert(std::abs(b.radius_sq - 1.0) < 1e-9);
    
    std::cout << "get_circum_ball passed!" << std::endl;
}

// Global state for callback testing
bool dummy_consents[] = {true, false, true, true};
int callback_count = 0;
bool test_consent_callback(int idx) {
    callback_count++;
    return dummy_consents[idx];
}

void test_welzl_consent() {
    std::cout << "Testing welzl_consent..." << std::endl;
    
    // Set up points: (0,0), (10,10) - non-consenting, (2,0), (0,2)
    // Consenting points: (0,0) [idx 0], (2,0) [idx 2], (0,2) [idx 3]
    // The circumsphere of these consenting points is center=(1,1), radius_sq = 2.0
    double points[] = {
        0.0, 0.0,
        10.0, 10.0,
        2.0, 0.0,
        0.0, 2.0
    };
    
    double center[2] = {0.0, 0.0};
    double radius_sq = -1.0;
    
    callback_count = 0;
    bool success = welzl_consent(points, 4, 2, test_consent_callback, center, &radius_sq);
    
    assert(success);
    assert(std::abs(center[0] - 1.0) < 1e-9);
    assert(std::abs(center[1] - 1.0) < 1e-9);
    assert(std::abs(radius_sq - 2.0) < 1e-9);
    // Point 1 (10,10) is non-consenting, so callback count should track
    assert(callback_count > 0);
    
    std::cout << "welzl_consent passed!" << std::endl;
}

int main() {
    test_solve_linear_system();
    test_get_circum_ball();
    test_welzl_consent();
    std::cout << "ALL C++ TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
