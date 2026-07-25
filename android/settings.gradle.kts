// The Android side of remapy. Deliberately a *separate* Gradle build rather than something
// wired into the pixi workspace: pixi is pinned to linux-64 and Python 3.14, and nothing in
// this directory is a Python dependency. The two builds share exactly one artifact, the
// goldens file that `pixi run export-fixtures` writes into `metrics/src/test/resources/`.

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

// Lets Gradle download the JDK the Android build needs instead of requiring one to be installed.
// Not a convenience: AGP needs a real `javac`, and a machine can easily have a JRE-only Java 21
// that runs Gradle and Kotlin perfectly well while providing no compiler at all. Provisioned JDKs
// land in ~/.gradle/jdks and touch nothing system-wide.
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "remapy-android"

// `metrics` is a plain Kotlin/JVM library on purpose — it has no Android dependency, so it
// compiles and its whole test suite runs on a desktop JDK with no SDK, emulator or device.
// That is what makes the equivalence harness cheap enough to run on every change; an
// androidTest-only kernel would need a device to tell you the maths is right.
include(":metrics")

// The camera, the detector and the UI. Everything that needs a device lives here, which is what
// keeps `:metrics` runnable on a plain JDK.
include(":app")
