"""Archive project files into SupplySense-AI.zip, excluding virtualenvs and temporary caches."""

import os
import zipfile


def make_zip(output_filename="SupplySense-AI.zip"):
    base_dir = os.path.abspath(os.getcwd())
    exclude_dirs = {
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".git",
        ".idea",
        ".vscode",
        ".ipynb_checkpoints",
    }
    exclude_extensions = {".pyc", ".pyo"}

    included_files = []
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file == output_filename or file.endswith(".zip"):
                    continue
                if any(file.endswith(ext) for ext in exclude_extensions):
                    continue
                # Exclude giant docx / pdf from report build if outside deliverables
                if file.endswith(".docx") or file.endswith(".pdf"):
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)

                # Store with prefix SupplySense-AI/
                archive_name = os.path.join("SupplySense-AI", rel_path)
                zipf.write(full_path, archive_name)
                included_files.append(archive_name)

    size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"Created {output_filename} successfully! Total files: {len(included_files)}, Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    make_zip()
