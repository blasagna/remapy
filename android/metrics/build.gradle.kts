import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
}

// No Android plugin and no `implementation` dependencies at all: the kernel is arrays and
// arithmetic, exactly as `motor_metrics` is numpy and two scipy functions. Anything that
// needs a Bitmap, a Context or a camera belongs in the app module, not here.
dependencies {
    testImplementation(libs.kotlin.test)
    testImplementation(libs.gson) // reads goldens.json; test-only, never shipped
}

// Bytecode target is pinned at 17 (what the Android toolchain consumes) but the build runs on
// whatever JDK is installed, rather than demanding a JDK 17 that may not be there. A
// `jvmToolchain(17)` would be stricter and is the usual advice, but it makes a checkout
// unbuildable on a machine with only a newer JDK, and this module has no JDK-specific code
// to protect.
//
// Deliberately NOT using `-Xjdk-release` / `options.release` to also pin the compiled-against
// API: those need a `ct.sym` that slimmer JDK builds do not ship, and this module's entire
// API surface is `kotlin.math` and arrays. If a real Android app module ever depends on this
// jar, that module's own AGP settings are what enforce API level, and they are stricter.
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_17
        // The never-mix rule and the NaN discipline both depend on warnings not being
        // ignorable noise.
        allWarningsAsErrors = true
    }
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("failed")
        showStandardStreams = false
    }
}
