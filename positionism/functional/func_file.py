def delete_lines(file_path, line_ranges):
    """
    Delete specified lines from a file.

    Args
    ====
    file_path: str
        The path to the file to be modified.
    line_ranges: list
        A list of tuples representing the ranges of lines to be deleted.

    Returns
    =======
    None
    """
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines_to_delete = set()
    for line_range in line_ranges:
        start, end = line_range
        if 1 <= start <= len(lines) and start <= end <= len(lines):
            lines_to_delete.update(range(start - 1, end))

    modified_lines = [line for i, line in enumerate(lines) if i not in lines_to_delete]

    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

    # print(f"Lines deleted successfully in file: {file_path}")


def delete_elements(file_path, line_ranges, line_numbers_edit, new_contents):
    """
    Delete specified lines and edit others in a file.

    Args
    ====
    file_path: str
        The path to the file to be modified.
    line_ranges: list
        A list of tuples representing the ranges of lines to be deleted.
    line_numbers_edit: list
        A list of line numbers to be edited.
    new_contents: list
        A list of new contents for the edited lines.

    Returns
    =======
    None
    """
    delete_lines(file_path, line_ranges)
    edit_lines(file_path, line_numbers_edit, new_contents)


def edit_lines(file_path, line_numbers, new_contents):
    """
    Edit specified lines in a file.

    Args
    ====
    file_path: str
        The path to the file to be modified.
    line_numbers: list
        A list of line numbers to be edited.
    new_contents: list
        A list of new contents for the edited lines.

    Returns
    =======
    None
    """
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify the line content
    for line_number, new_content in zip(line_numbers, new_contents):
        if 1 <= line_number <= len(lines):
            lines[line_number - 1] = new_content + '\n'

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
            # # print(f"Line edited successfully.")
    # # else:
    # # #     print(f"Invalid line number: {line_number}")
