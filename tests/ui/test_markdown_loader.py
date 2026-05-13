#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Single comprehensive test for markdown_loader — covers all transformation types."""

from cutana_ui.utils.markdown_loader import format_markdown_display, get_markdown_content


def test_get_markdown_content_missing_file():
    """Loading a nonexistent file returns error message."""
    result = get_markdown_content("/nonexistent/file.md")
    assert "not found" in result.lower() or "error" in result.lower()


def test_format_markdown_display_comprehensive():
    """One test exercising all major markdown transformations."""
    md = """\
# Main Title
## Subtitle

This has **bold**, *italic*, and `inline code`.

- List item 1
- List item 2

1. Ordered item
2. Another item

> A blockquote

[A link](https://example.com)

---

```python
def hello():
    return "world"
```

![alt text](image.png)

[//]: # (This is a comment)

Plain paragraph text with &amp; entities.
"""
    html = format_markdown_display(md)

    # Headers
    assert "md_display_h1" in html
    assert "Main Title" in html

    # Inline formatting
    assert "md_display_strong" in html
    assert "md_display_em" in html
    assert "md_display_code" in html

    # Lists
    assert "md_display_li" in html

    # Blockquote
    assert "md_display_blockquote" in html

    # Link
    assert "example.com" in html

    # Code block with syntax highlighting
    assert "hello" in html
    assert "language-python" in html

    # Image
    assert "md_display_img" in html

    # Overall structure
    assert "md_display_container" in html
