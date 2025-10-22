# PhysGrad Manipulation Examples

# Manipulation Demo 1: Gradient-Based Pushing
add_executable(demo_pushing_optimization
    manipulation/demo_pushing_optimization.cpp
)

target_link_libraries(demo_pushing_optimization
    physgrad_core
    pthread
)

# Set C++ standard
set_target_properties(demo_pushing_optimization PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Installation
install(TARGETS demo_pushing_optimization
    RUNTIME DESTINATION bin/examples
)

# Future demos can be added here
# add_executable(demo_grasping ...)
# add_executable(demo_stacking ...)
