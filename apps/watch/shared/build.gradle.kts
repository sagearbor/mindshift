plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.android.library)
}

kotlin {
    jvm()
    androidTarget()

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(libs.kotlinx.serialization.json)
                implementation(libs.kotlinx.coroutines.core)
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
    }
}

android {
    namespace = "app.gauge.shared"
    compileSdk = 34

    defaultConfig {
        // Lowered 30 -> 26 for the phone track (Plan 2b Task 1): androidApp targets minSdk 26,
        // and an app depending on a library with a higher minSdk fails manifest merge. Safe for
        // wearApp, which declares its own minSdk 30 (>= 26) independently.
        minSdk = 26
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
