import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

// Upload-key signing props: gitignored keystore.properties at the repo root
// (keystore itself lives outside the repo in ~/.config/gauge/). Loaded at the
// top level because `java.*` is shadowed inside the android {} block.
val keystoreProps = Properties().apply {
    val f = rootProject.file("keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}
val hasUploadKey = keystoreProps.containsKey("storeFile")

android {
    namespace = "app.gauge.wear"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.sagearbor.gauge.wear"
        minSdk = 30
        targetSdk = 34
        versionCode = 14
        versionName = "0.4.4"

        buildConfigField(
            "String",
            "GAUGE_API_BASE",
            "\"https://mindshift-api-664594784582.us-central1.run.app\"",
        )
        buildConfigField("boolean", "TELEMETRY_ENABLED", "true")
    }

    // Play rejects debug-mode signatures even on the internal track, so release
    // builds MUST use the upload key (props loaded at top of file). Falls back
    // to debug signing only when keystore.properties is absent (fresh clones).
    signingConfigs {
        if (hasUploadKey) {
            create("upload") {
                storeFile = file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = if (hasUploadKey) signingConfigs.getByName("upload")
                            else signingConfigs.getByName("debug")
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.11"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation(project(":shared"))

    implementation(platform(libs.compose.bom))
    implementation(libs.wear.compose.material)
    implementation(libs.wear.compose.foundation)
    implementation(libs.wear.compose.navigation)
    implementation(libs.activity.compose)
    implementation(libs.ui.tooling.preview)
    debugImplementation(libs.ui.tooling)

    implementation(libs.okhttp)
    implementation(libs.tiles)
    implementation(libs.tiles.material)
    implementation(libs.protolayout)
    implementation(libs.watchface.complications.data.source.ktx)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.health.services.client)
    implementation(libs.guava) // see libs.versions.toml comment: compile-time guava/listenablefuture fix

    testImplementation(kotlin("test"))
    testImplementation(libs.okhttp.mockwebserver)
    testImplementation(libs.kotlinx.coroutines.test)
}
