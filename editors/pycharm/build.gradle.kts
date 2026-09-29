import org.jetbrains.intellij.platform.gradle.TestFrameworkType
import org.jetbrains.intellij.platform.gradle.IntelliJPlatformType
import org.jetbrains.intellij.platform.gradle.tasks.PrepareSandboxTask
import org.jetbrains.intellij.platform.gradle.tasks.VerifyPluginTask
import org.jetbrains.changelog.markdownToHTML
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.gradle.api.tasks.compile.JavaCompile

plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.intellij.platform")
    id("org.jetbrains.changelog")
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_21)
    }
}

tasks.withType<JavaCompile>().configureEach {
    sourceCompatibility = JavaVersion.VERSION_21.toString()
    targetCompatibility = JavaVersion.VERSION_21.toString()
    options.release.set(21)
}

dependencies {
    testImplementation("junit:junit:4.13.2")

    // IntelliJ Platform Gradle Plugin Dependencies Extension - read more: https://plugins.jetbrains.com/docs/intellij/tools-intellij-platform-gradle-plugin-dependencies-extension.html
    intellijPlatform {
        val pycharmVersion = providers.gradleProperty("pycharmVersion").get()
        val pycharmHome = providers.environmentVariable("PYCHARM_HOME").orNull
        if (pycharmHome == null) {
            pycharm(pycharmVersion)
        } else {
            local(pycharmHome)
        }
        bundledPlugin("PythonCore")
        testFramework(TestFrameworkType.Platform)
    }
}

intellijPlatform {
    pluginVerification {
        ides {
            create(IntelliJPlatformType.PyCharm, "2026.1.4")
        }
    }

    pluginConfiguration {
        description = providers.fileContents(layout.projectDirectory.file("README.md")).asText.map { readme ->
            val start = "<!-- Plugin description -->"
            val end = "<!-- Plugin description end -->"
            val startIndex = readme.indexOf(start)
            val endIndex = readme.indexOf(end)
            if (startIndex < 0 || endIndex < startIndex) {
                throw GradleException("Plugin description section not found in README.md")
            }
            markdownToHTML(readme.substring(startIndex + start.length, endIndex).trim())
        }
        ideaVersion {
            // The non-deprecated LSP client API and integration provider require 2026.1.4+.
            sinceBuild = "261.1"
            untilBuild = provider { null }
        }
    }
}

tasks.named<VerifyPluginTask>("verifyPlugin") {
    dependsOn(tasks.named("buildPlugin"))
}

tasks.named<PrepareSandboxTask>("prepareSandbox") {
    from(files("README.md", "LICENSE")) {
        into(pluginName)
    }
}
