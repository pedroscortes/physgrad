# PhysGrad Manipulation Examples

# Manipulation Demo 1: Gradient-Based Pushing
add_executable(demo_pushing_optimization
    manipulation/demo_pushing_optimization.cpp
)

target_link_libraries(demo_pushing_optimization
    physgrad_core
    pthread
)

set_target_properties(demo_pushing_optimization PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Manipulation Demo 2: Grasping Optimization
add_executable(demo_grasping_optimization
    manipulation/demo_grasping_optimization.cpp
)

target_link_libraries(demo_grasping_optimization
    physgrad_core
    pthread
)

set_target_properties(demo_grasping_optimization PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Manipulation Demo 3: Object Stacking
add_executable(demo_stacking_optimization
    manipulation/demo_stacking_optimization.cpp
)

target_link_libraries(demo_stacking_optimization
    physgrad_core
    pthread
)

set_target_properties(demo_stacking_optimization PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Installation
install(TARGETS
    demo_pushing_optimization
    demo_grasping_optimization
    demo_stacking_optimization
    RUNTIME DESTINATION bin/examples
)
