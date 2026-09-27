package dev.shpe.pycharm

import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.platform.lsp.api.LspServerSupportProvider
import com.intellij.platform.lsp.api.ProjectWideLspServerDescriptor
import java.nio.file.Files
import java.nio.file.Path

internal class ShpeLspServerSupportProvider : LspServerSupportProvider {
    override fun fileOpened(
        project: Project,
        file: VirtualFile,
        serverStarter: LspServerSupportProvider.LspServerStarter,
    ) {
        if (file.extension.equals("py", ignoreCase = true)) {
            serverStarter.ensureServerStarted(ShpeLspServerDescriptor(project))
        }
    }
}

private class ShpeLspServerDescriptor(project: Project) :
    ProjectWideLspServerDescriptor(project, "Shpe") {
    private val projectBasePath = project.basePath

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

        return GeneralCommandLine(python, "-m", "shpe.server").apply {
            projectPath?.let { withWorkDirectory(it.toString()) }
        }
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
