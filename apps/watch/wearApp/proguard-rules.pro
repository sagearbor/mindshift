# R8 rules for the MindShift Wear OS app (release builds only).
#
# Library rules (Compose, Wear Tiles/ProtoLayout, complications, OkHttp,
# kotlinx.coroutines, kotlinx.serialization) ship as consumer rules inside
# their artifacts. This file only covers what those cannot know about.

# Our own code is small next to the libraries and there is no on-device test
# loop (installs go through Play, no adb), so it is kept whole: R8 still strips
# and optimises the libraries, which is where the 26 MB came from. This rules
# out any class-name / reflection surprise in the services, the tile, the
# complication and the @Serializable wire models.
-keep class app.gauge.** { *; }

# kotlinx.serialization (belt-and-braces over its bundled rules).
-keepattributes *Annotation*, InnerClasses, Signature, EnclosingMethod
-dontnote kotlinx.serialization.**
-keepclassmembers @kotlinx.serialization.Serializable class ** {
    static ** Companion;
    *** Companion;
    kotlinx.serialization.KSerializer serializer(...);
}
-keepclasseswithmembers class **$$serializer { *; }

# OkHttp / Okio optional platform integrations that are not on Android.
-dontwarn okhttp3.internal.platform.**
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**

# guava (pulled in for ListenableFuture) references annotations and JDK-only
# classes that are not on the Android classpath.
-dontwarn com.google.errorprone.annotations.**
-dontwarn com.google.j2objc.annotations.**
-dontwarn javax.annotation.**
-dontwarn org.checkerframework.**
-dontwarn afu.org.checkerframework.**
-dontwarn java.lang.reflect.AnnotatedType
-dontwarn sun.misc.Unsafe

# Keep line numbers so Play vitals / telemetry stack traces stay readable
# (upload build/outputs/mapping/release/mapping.txt with the AAB).
-keepattributes SourceFile, LineNumberTable
-renamesourcefileattribute SourceFile
