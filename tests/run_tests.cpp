#include "test_framework.h"
#include <iostream>
#include <string>
#include <vector>

using namespace physgrad_tests;

// Test function declarations
void testDifferentiableContact(TestFramework& framework);
void testRigidBody(TestFramework& framework);
void testSymplecticIntegrators(TestFramework& framework);
void testConstraints(TestFramework& framework);
void testCollisionDetection(TestFramework& framework);

struct TestSuite {
    std::string name;
    std::function<void(TestFramework&)> test_function;
};

int main(int argc, char* argv[]) {
    std::cout << "PhysGrad Test Suite" << std::endl;
    std::cout << "==================" << std::endl;

    TestFramework framework;
    bool verbose = false;
    std::vector<std::string> selected_suites;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options] [test_suites...]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --verbose, -v    Enable verbose output" << std::endl;
            std::cout << "  --help, -h       Show this help message" << std::endl;
            std::cout << "Test Suites:" << std::endl;
            std::cout << "  contact          Differentiable contact mechanics tests" << std::endl;
            std::cout << "  rigid_body       Rigid body dynamics tests" << std::endl;
            std::cout << "  symplectic       Symplectic integrator tests" << std::endl;
            std::cout << "  constraints      Constraint-based physics tests" << std::endl;
            std::cout << "  collision        Collision detection tests" << std::endl;
            std::cout << "  all              Run all test suites (default)" << std::endl;
            return 0;
        } else {
            selected_suites.push_back(arg);
        }
    }

    framework.setVerbose(verbose);

    // Define all test suites
    std::vector<TestSuite> test_suites = {
        {"contact", testDifferentiableContact},
        {"rigid_body", testRigidBody},
        {"symplectic", testSymplecticIntegrators},
        {"constraints", testConstraints},
        {"collision", testCollisionDetection}
    };

    // Determine which test suites to run
    std::vector<TestSuite> suites_to_run;
    if (selected_suites.empty() || (selected_suites.size() == 1 && selected_suites[0] == "all")) {
        suites_to_run = test_suites;
    } else {
        for (const std::string& suite_name : selected_suites) {
            bool found = false;
            for (const auto& suite : test_suites) {
                if (suite.name == suite_name) {
                    suites_to_run.push_back(suite);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Unknown test suite: " << suite_name << std::endl;
                return 1;
            }
        }
    }

    // Run selected test suites
    std::cout << "\nRunning " << suites_to_run.size() << " test suite(s)..." << std::endl;

    for (const auto& suite : suites_to_run) {
        std::cout << "\n--- Running " << suite.name << " tests ---" << std::endl;
        try {
            suite.test_function(framework);
            std::cout << "✓ " << suite.name << " tests completed" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "✗ " << suite.name << " tests failed with exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "✗ " << suite.name << " tests failed with unknown exception" << std::endl;
        }
    }

    // Print summary
    framework.printSummary();

    // Return appropriate exit code
    return framework.allTestsPassed() ? 0 : 1;
}