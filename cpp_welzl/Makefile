CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17

all: libwelzl.so test_welzl

libwelzl.so: welzl.cpp welzl.hpp
	$(CXX) $(CXXFLAGS) -fPIC -shared -o libwelzl.so welzl.cpp

test_welzl: test_welzl.cpp welzl.cpp welzl.hpp
	$(CXX) $(CXXFLAGS) -o test_welzl test_welzl.cpp welzl.cpp

run_tests: test_welzl
	./test_welzl

clean:
	rm -f libwelzl.so test_welzl

.PHONY: all run_tests clean
