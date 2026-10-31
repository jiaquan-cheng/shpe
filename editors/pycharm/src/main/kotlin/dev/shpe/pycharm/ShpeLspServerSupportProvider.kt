package dev.shpe.pycharm

import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.ide.plugins.PluginManagerCore
import com.intellij.notification.Notification
import com.intellij.notification.NotificationType
import com.intellij.openapi.extensions.PluginId
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.platform.lsp.api.LspIntegrationProvider
import com.intellij.platform.lsp.api.ProjectWideLspClientDescriptor
import com.intellij.util.concurrency.AppExecutorUtil
import com.intellij.util.text.VersionComparatorUtil
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

internal class ShpeLspIntegrationProvider : LspIntegrationProvider {
    override fun fileOpened(
        project: Project,
        file: VirtualFile,
        clientStarter: LspIntegrationProvider.LspClientStarter,
    ) {
        if (file.extension.equals("py", ignoreCase = true)) {
            clientStarter.ensureClientStarted(ShpeLspClientDescriptor(project))
        }
    }
}

private class ShpeLspClientDescriptor(project: Project) :
    ProjectWideLspClientDescriptor(project, "Shpe") {
    private val projectBasePath = project.basePath
    private var packageVersionChecked = false

    override fun isSupportedFile(file: VirtualFile): Boolean =
        file.extension.equals("py", ignoreCase = true)

    override fun createCommandLine(): GeneralCommandLine {
        val projectPath = projectBasePath?.let { Path.of(it) }
        val python = System.getenv("SHPE_PYTHON")
            ?: projectPath?.let(::projectVirtualEnvPython)
            ?: if (System.getProperty("os.name").startsWith("Windows", ignoreCase = true)) {
                "python"
            } else {
                "python3"
            }

        checkPythonPackageVersion(python)

        return GeneralCommandLine(python, "-m", "shpe.server").apply {
            projectPath?.let { withWorkDirectory(it.toString()) }
        }
    }

    @Synchronized
    private fun checkPythonPackageVersion(python: String) {
        if (packageVersionChecked) return
        packageVersionChecked = true

        val pluginVersion = PluginManagerCore.getPlugin(PluginId.getId("dev.shpe"))?.version ?: return
        val process = try {
            ProcessBuilder(
                python,
                "-c",
                "import importlib.metadata; import shpe; print(importlib.metadata.version('shpe'))",
            ).redirectErrorStream(true).start()
        } catch (_: Exception) {
            null
        }

        if (process == null) {
            notifyPackageWarning(
                "The 'shpe' package could not be checked for '$python'. Install or update it with " +
                    "'$python -m pip install --upgrade shpe'.",
            )
            return
        }

        AppExecutorUtil.getAppExecutorService().execute {
            var checkTimedOut = false
            val installedVersion = try {
                if (!process.waitFor(5, TimeUnit.SECONDS)) {
                    checkTimedOut = true
                    process.destroyForcibly()
                    null
                } else if (process.exitValue() == 0) {
                    process.inputStream.bufferedReader().use { it.readText().trim() }
                        .takeIf(String::isNotEmpty)
                } else {
                    null
                }
            } catch (_: Exception) {
                null
            }

            val message = when {
                checkTimedOut ->
                    "The 'shpe' package version check timed out for '$python'. Confirm the environment has Shpe installed " +
                        "and update it with '$python -m pip install --upgrade shpe'."
                installedVersion == null ->
                    "The 'shpe' package was not found for '$python'. Install it with '$python -m pip install shpe'."
                VersionComparatorUtil.compare(installedVersion, pluginVersion) < 0 ->
                    "The installed 'shpe' package version ($installedVersion) is older than this PyCharm plugin version " +
                        "($pluginVersion). Update it with '$python -m pip install --upgrade shpe'."
                else -> null
            } ?: return@execute

            notifyPackageWarning(message)
        }
    }

    private fun notifyPackageWarning(message: String) {
        if (project.isDisposed) return
        Notification("Shpe", "Shpe Python package", message, NotificationType.WARNING)
            .notify(project)
    }

    private fun projectVirtualEnvPython(projectPath: Path): String? {
        val relativePath = if (System.getProperty("os.name").startsWith("Windows", ignoreCase = true)) {
            Path.of(".venv", "Scripts", "python.exe")
        } else {
            Path.of(".venv", "bin", "python")
        }
        val interpreter = projectPath.resolve(relativePath)
        return interpreter.takeIf(Files::isExecutable)?.toString()
    }
}
