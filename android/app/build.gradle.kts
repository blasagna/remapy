// Imported rather than fully qualified: `java` in a build script resolves to Gradle's `java`
// extension, which shadows the package name.
import java.net.URI

// No `org.jetbrains.kotlin.android`: AGP 9 has built-in Kotlin support and rejects the separate
// plugin. `:metrics` still uses the standalone `kotlin.jvm` plugin, because it is a plain JVM
// library with no AGP anywhere near it.
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "dev.remapy.app"
    compileSdk = 37

    defaultConfig {
        applicationId = "dev.remapy.app"
        // 26 covers everything MediaPipe Tasks and CameraX need, and anything older is not a
        // phone anyone is holding at a therapy session in 2026.
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "0.1"

        ndk {
            // MediaPipe's native library is ~10-15 MB *per ABI*, and shipping all four makes a
            // 65 MB debug APK. arm64-v8a is every phone this will run on; x86_64 is kept so the
            // build is installable on an emulator.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    androidResources {
        // The model bundles are already compressed; letting aapt re-compress them costs build
        // time and, worse, forces MediaPipe to extract them to a temp file at startup instead of
        // mapping them straight out of the APK.
        noCompress += listOf("task", "tflite")
    }
}

// Pinned rather than inherited from the Gradle JVM, so the build does not depend on which Java
// happens to be installed — and so it works at all on a machine whose only Java is a JRE.
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}

dependencies {
    implementation(project(":metrics"))
    implementation(libs.mediapipe.tasks.vision)

    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
}

// --------------------------------------------------------------------------- //
// Model bundles
// --------------------------------------------------------------------------- //

/**
 * Put the MediaPipe model bundles in `assets/`, preferring the copies the Python side has
 * already cached.
 *
 * This mirrors `pose_estimation/model.py` and `face_blur/model.py`: same URLs, same
 * download-on-first-use, same "not checked in" treatment (the Python `models/` directories are
 * gitignored and so is this one). Copying from the sibling cache when it exists is not just a
 * speed-up — it guarantees the phone and the laptop are running the *same bytes*, which is one
 * fewer variable in the landmark-parity comparison.
 */
val modelSources = listOf(
    Triple(
        "pose_landmarker_lite.task",
        rootProject.file("../pose_estimation/models/pose_landmarker_lite.task"),
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/" +
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ),
    Triple(
        "blaze_face_short_range.tflite",
        rootProject.file("../face_blur/models/blaze_face_short_range.tflite"),
        "https://storage.googleapis.com/mediapipe-models/face_detector/" +
            "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    ),
)

val fetchModels = tasks.register("fetchModels") {
    val assetsDir = layout.projectDirectory.dir("src/main/assets")
    outputs.dir(assetsDir)
    doLast {
        val dir = assetsDir.asFile
        dir.mkdirs()
        for ((name, cached, url) in modelSources) {
            val target = dir.resolve(name)
            if (target.exists() && target.length() > 0) continue
            if (cached.exists()) {
                logger.lifecycle("Copying $name from the Python model cache")
                cached.copyTo(target, overwrite = true)
            } else {
                logger.lifecycle("Downloading $name from $url")
                val part = dir.resolve("$name.part")
                URI(url).toURL().openStream().use { input ->
                    part.outputStream().use { output -> input.copyTo(output) }
                }
                part.renameTo(target)
            }
        }
    }
}

tasks.named("preBuild") { dependsOn(fetchModels) }
