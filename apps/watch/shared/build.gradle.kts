plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.android.library)
}

// Track 1 (2026-08-24): cross-language golden vectors. The ONE source of truth
// for the nudge policy's behaviour is server/tests/fixtures/policy_vectors/
// (see its README) — the Python NudgePolicy is the reference and the Kotlin
// NudgeStateMachine is its offline mirror, and both must replay the same JSON.
// Rather than hand-copying that JSON into this module's test resources (where
// it would silently rot the first time someone edited the server copy), this
// Sync task copies the canonical directory into a build-generated resource
// root at test time, and jvmTest's resources include that root. Nobody edits
// a copy; NudgeStateMachineVectorsTest.kt reads /policy_vectors/*.json off
// the test classpath and gets whatever the server side currently says.
//
// `Sync` (not `Copy`) so a case file deleted upstream disappears here too.
// The relative path is from the watch build's ROOT (apps/watch) up to the repo
// root — the watch build is a standalone Gradle project inside the monorepo,
// so there is no Gradle-native way to reference server/ other than by path.
// (rootProject, not this module's own directory: apps/watch/shared/../../ is
// apps/, which is one level short — an easy silent miss because Sync of a
// nonexistent directory is a no-op, not an error.)
val policyVectorsSource = rootProject.layout.projectDirectory.dir("../../server/tests/fixtures/policy_vectors")
val policyVectorsRoot = layout.buildDirectory.dir("generated/policy-vectors")
val syncPolicyVectors by tasks.registering(Sync::class) {
    description = "Copies the canonical policy_vectors/*.json fixtures from server/tests into the shared JVM test resources."
    from(policyVectorsSource) {
        include("*.json")
        into("policy_vectors")
    }
    into(policyVectorsRoot)
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
        // The vector test lives in jvmTest (not commonTest) because reading a
        // file off the classpath is a JVM affordance — Kotlin common has no
        // resource API. jvm() is exactly the target :shared:allTests runs, so
        // the contract is enforced on every `allTests` run.
        val jvmTest by getting {
            resources.srcDir(policyVectorsRoot)
        }
    }
}

// KMP's jvm-target resource processing doesn't infer a task dependency from a
// srcDir the way a plain JVM source set would (verified 2026-08-24: with only
// `resources.srcDir(syncPolicyVectors)` the Sync never ran and the test failed
// with "missing from the test classpath"), so wire it explicitly.
tasks.named("jvmTestProcessResources") {
    dependsOn(syncPolicyVectors)
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
