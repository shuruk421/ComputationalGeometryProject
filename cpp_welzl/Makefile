CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17

all: libwelzl.so test_welzl welzl_cpp_module.so

libwelzl.so: welzl.cpp welzl.hpp
	$(CXX) $(CXXFLAGS) -fPIC -shared -o libwelzl.so welzl.cpp

test_welzl: test_welzl.cpp welzl.cpp welzl.hpp
	$(CXX) $(CXXFLAGS) -o test_welzl test_welzl.cpp welzl.cpp

welzl_cpp_module.so: welzl_pybind.cpp welzl.cpp welzl.hpp
	$(CXX) $(CXXFLAGS) -fPIC -shared $(shell python3 -m pybind11 --includes) -o welzl_cpp_module.so welzl_pybind.cpp welzl.cpp

run_tests: test_welzl
	./test_welzl

clean:
	rm -f libwelzl.so test_welzl welzl_cpp_module.so

.PHONY: all run_tests clean
