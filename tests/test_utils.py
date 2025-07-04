"""Unit tests for docxtract.utils module."""

import tempfile
from pathlib import Path

import pytest

from docxtract.utils import ensure_directory, get_default_output_path, validate_pdf_file


class TestUtilsFunctions:
    """Test cases for utility functions."""

    def test_ensure_directory_creates_new_directory(self):
        """Test ensure_directory creates a new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_directory"

            # Directory should not exist initially
            assert not new_dir.exists()

            ensure_directory(new_dir)

            # Directory should exist after calling ensure_directory
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_ensure_directory_creates_nested_directories(self):
        """Test ensure_directory creates nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "level1" / "level2" / "level3"

            # Nested directories should not exist initially
            assert not nested_dir.exists()
            assert not nested_dir.parent.exists()

            ensure_directory(nested_dir)

            # All nested directories should exist
            assert nested_dir.exists()
            assert nested_dir.is_dir()
            assert nested_dir.parent.exists()
            assert nested_dir.parent.parent.exists()

    def test_ensure_directory_existing_directory(self):
        """Test ensure_directory with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir) / "existing"
            existing_dir.mkdir()

            # Directory should exist
            assert existing_dir.exists()

            # Should not raise error with existing directory
            ensure_directory(existing_dir)

            # Directory should still exist
            assert existing_dir.exists()
            assert existing_dir.is_dir()

    def test_ensure_directory_with_file_path(self):
        """Test ensure_directory with a file path (should create parent)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "subdir" / "file.txt"
            parent_dir = file_path.parent

            # Parent directory should not exist initially
            assert not parent_dir.exists()

            ensure_directory(parent_dir)

            # Parent directory should exist
            assert parent_dir.exists()
            assert parent_dir.is_dir()

    def test_validate_pdf_file_valid_pdf(self):
        """Test validate_pdf_file with valid PDF file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "test.pdf"

            # Create a dummy PDF file
            pdf_path.write_bytes(
                b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF"
            )

            # Should validate as true
            assert validate_pdf_file(pdf_path) is True

    def test_validate_pdf_file_nonexistent_file(self):
        """Test validate_pdf_file with non-existent file."""
        nonexistent_path = Path("/path/that/does/not/exist.pdf")

        # Should validate as false
        assert validate_pdf_file(nonexistent_path) is False

    def test_validate_pdf_file_directory(self):
        """Test validate_pdf_file with directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)

            # Should validate as false (it's a directory, not a file)
            assert validate_pdf_file(dir_path) is False

    def test_validate_pdf_file_wrong_extension(self):
        """Test validate_pdf_file with wrong file extension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_path = Path(temp_dir) / "test.txt"
            txt_path.write_text("This is not a PDF file")

            # Should validate as false (wrong extension)
            assert validate_pdf_file(txt_path) is False

    def test_validate_pdf_file_case_insensitive_extension(self):
        """Test validate_pdf_file with case variations of PDF extension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different case variations
            for extension in [".PDF", ".Pdf", ".pDf"]:
                pdf_path = Path(temp_dir) / f"test{extension}"
                pdf_path.write_bytes(b"dummy pdf content")

                # Should validate as true (case insensitive)
                assert validate_pdf_file(pdf_path) is True

    def test_validate_pdf_file_no_extension(self):
        """Test validate_pdf_file with file without extension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            no_ext_path = Path(temp_dir) / "filename_without_extension"
            no_ext_path.write_text("Some content")

            # Should validate as false (no PDF extension)
            assert validate_pdf_file(no_ext_path) is False

    def test_get_default_output_path_basic(self):
        """Test get_default_output_path with basic PDF file."""
        input_path = Path("/path/to/document.pdf")
        expected_output = Path("/path/to/document.md")

        result = get_default_output_path(input_path)

        assert result == expected_output

    def test_get_default_output_path_different_directory(self):
        """Test get_default_output_path with different directory structure."""
        input_path = Path("/home/user/papers/research_paper.pdf")
        expected_output = Path("/home/user/papers/research_paper.md")

        result = get_default_output_path(input_path)

        assert result == expected_output

    def test_get_default_output_path_relative_path(self):
        """Test get_default_output_path with relative path."""
        input_path = Path("./documents/paper.pdf")
        expected_output = Path("./documents/paper.md")

        result = get_default_output_path(input_path)

        assert result == expected_output

    def test_get_default_output_path_no_extension(self):
        """Test get_default_output_path with file without extension."""
        input_path = Path("/path/to/document")
        expected_output = Path("/path/to/document.md")

        result = get_default_output_path(input_path)

        assert result == expected_output

    def test_get_default_output_path_multiple_dots(self):
        """Test get_default_output_path with filename containing multiple dots."""
        input_path = Path("/path/to/document.v1.2.pdf")
        expected_output = Path("/path/to/document.v1.2.md")

        result = get_default_output_path(input_path)

        assert result == expected_output

    def test_get_default_output_path_uppercase_extension(self):
        """Test get_default_output_path with uppercase extension."""
        input_path = Path("/path/to/document.PDF")
        expected_output = Path("/path/to/document.md")

        result = get_default_output_path(input_path)

        assert result == expected_output

    def test_get_default_output_path_preserves_directory_structure(self):
        """Test that get_default_output_path preserves the directory structure."""
        input_path = Path("/very/deep/nested/directory/structure/file.pdf")
        result = get_default_output_path(input_path)

        # Should preserve the directory structure
        assert result.parent == input_path.parent
        assert result.name == "file.md"

    @pytest.mark.parametrize(
        "input_filename,expected_output",
        [
            ("simple.pdf", "simple.md"),
            ("with spaces.pdf", "with spaces.md"),
            ("with-hyphens.pdf", "with-hyphens.md"),
            ("with_underscores.pdf", "with_underscores.md"),
            ("UPPERCASE.pdf", "UPPERCASE.md"),
            ("MixedCase.pdf", "MixedCase.md"),
            ("numbers123.pdf", "numbers123.md"),
            ("special!@#.pdf", "special!@#.md"),
        ],
    )
    def test_get_default_output_path_various_filenames(
        self, input_filename, expected_output
    ):
        """Test get_default_output_path with various filename patterns."""
        input_path = Path(f"/test/{input_filename}")
        expected_path = Path(f"/test/{expected_output}")

        result = get_default_output_path(input_path)

        assert result == expected_path

    def test_functions_work_with_pathlib_objects(self):
        """Test that all functions properly work with pathlib.Path objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with Path objects
            base_path = Path(temp_dir)
            test_dir = base_path / "test_dir"
            test_file = base_path / "test.pdf"

            # Create test file
            test_file.write_bytes(b"dummy pdf content")

            # All functions should work with Path objects
            ensure_directory(test_dir)
            assert test_dir.exists()

            is_valid = validate_pdf_file(test_file)
            assert is_valid is True

            output_path = get_default_output_path(test_file)
            assert isinstance(output_path, Path)
            assert output_path.suffix == ".md"
