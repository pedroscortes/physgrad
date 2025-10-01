#pragma once

#include <type_traits>

// Forward declarations for physics concepts to avoid circular dependencies
// NOTE: These are simplified forward declarations only - full definitions are in physics_concepts.h
namespace physgrad::concepts::detail {
    // Simplified forward declarations to break circular dependencies
    template<typename T>
    concept SimplePhysicsScalar = std::floating_point<T>;

    template<typename T>
    concept SimpleGPUCompatible = std::is_trivially_copyable_v<T>;

    template<typename T>
    concept SimpleVector3D = requires { typename T::value_type; };

    template<typename T>
    concept SimpleParticle = requires { typename T::scalar_type; };

    template<typename T>
    concept SimpleIntegrator = requires { typename T::scalar_type; };
}

// Namespace alias for compatibility while avoiding redefinition
namespace physgrad::concepts {
    namespace fwd = detail;
}