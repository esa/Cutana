#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Markdown loading and rendering utilities for Cutana UI."""

import os
from loguru import logger


def get_markdown_content(markdown_path):
    """
    Load raw markdown content from a file.

    Args:
        markdown_path (str): Full path to the markdown file

    Returns:
        str: Raw markdown content
    """
    try:
        if not os.path.exists(markdown_path):
            logger.error(f"Markdown file not found: {markdown_path}")
            return f"Markdown file not found: {markdown_path}"

        with open(markdown_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        return md_content
    except Exception as e:
        logger.error(f"Error loading markdown file: {e}")
        return f"Error loading markdown: {str(e)}"


def format_markdown_display(md_content):
    """
    Format markdown content with basic Markdown-to-HTML conversion
    according to standard Markdown syntax.

    Args:
        md_content (str): Raw markdown content

    Returns:
        str: HTML with formatted markdown for display
    """
    import re

    # Debug: Print the input content length
    logger.debug(f"Formatting markdown content with length: {len(md_content)}")

    # Helper function to escape HTML special characters
    def escape_html(text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Process code blocks first (``` or ~~~)
    code_blocks = []
    # Use a clearer placeholder to temporarily replace code blocks
    placeholder_pattern = "CODEBLOCKPLACEHOLDER{}CODEBLOCK_"

    # More robust regex pattern for code blocks - handles various types of delimiters and whitespace
    code_block_pattern = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)

    # Process and replace code blocks
    def replace_code_blocks(match):
        lang = match.group(1).strip() if match.group(1) else ""
        code = match.group(2) if match.group(2) else ""
        idx = len(code_blocks)
        placeholder = placeholder_pattern.format(idx)
        code_blocks.append((lang, code))
        logger.debug(f"Found code block {idx} with language '{lang}' and length {len(code)}")
        return placeholder

    # Replace code blocks with placeholders
    content = code_block_pattern.sub(replace_code_blocks, md_content)

    # Define formatting patterns
    inline_code_pattern = re.compile(r"`([^`]+?)`")
    bold_pattern = re.compile(r"\*\*([^*]+?)\*\*")
    italic_pattern = re.compile(r"\*([^*]+?)\*")

    # Split the content into lines for processing
    lines = content.split("\n")
    processed_lines = []

    # Keep track of list context
    in_ordered_list = False
    in_unordered_list = False
    list_indent_level = 0
    current_list_number = 0

    for line in lines:
        # Handle headers
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            level = len(header_match.group(1))
            text = header_match.group(2)
            # Process formatting in headers
            text = bold_pattern.sub(r"<strong>\1</strong>", text)
            text = inline_code_pattern.sub(
                r'<code style="background: #353b45; padding: 2px 5px; border-radius: 3px; font-family: monospace; color: #e5c07b;">\1</code>',
                text,
            )
            text = (
                escape_html(text)
                .replace("&lt;strong&gt;", "<strong>")
                .replace("&lt;/strong&gt;", "</strong>")
            )
            text = text.replace("&lt;code ", "<code ").replace("&lt;/code&gt;", "</code>")
            processed_lines.append(f"<h{level}>{text}</h{level}>")
            # Reset list tracking
            in_ordered_list = False
            in_unordered_list = False
            continue

        # Handle horizontal rules
        if re.match(r"^(\*\*\*|\-\-\-|\_\_\_)$", line.strip()):
            processed_lines.append("<hr>")
            # Reset list tracking
            in_ordered_list = False
            in_unordered_list = False
            continue

        # Handle blockquotes
        blockquote_match = re.match(r"^>\s+(.+)$", line)
        if blockquote_match:
            text = blockquote_match.group(1)
            # Process formatting in blockquotes
            text = bold_pattern.sub(r"<strong>\1</strong>", text)
            text = inline_code_pattern.sub(
                r'<code style="background: #353b45; padding: 2px 5px; border-radius: 3px; font-family: monospace; color: #e5c07b;">\1</code>',
                text,
            )
            text = (
                escape_html(text)
                .replace("&lt;strong&gt;", "<strong>")
                .replace("&lt;/strong&gt;", "</strong>")
            )
            text = text.replace("&lt;code ", "<code ").replace("&lt;/code&gt;", "</code>")
            processed_lines.append(f"<blockquote>{text}</blockquote>")
            # Reset list tracking
            in_ordered_list = False
            in_unordered_list = False
            continue

        # Handle blockquotes
        blockquote_match = re.match(r"^>\s+(.+)$", line)
        if blockquote_match:
            text = escape_html(blockquote_match.group(1))
            processed_lines.append(f"<blockquote>{text}</blockquote>")
            continue

        # Handle unordered lists
        list_match = re.match(r"^(\s*)[\*\-\+]\s+(.+)$", line)
        if list_match:
            indent = len(list_match.group(1))
            text = list_match.group(2)

            # Process inline code first - important to handle this before escaping HTML
            text = inline_code_pattern.sub(
                r'<code style="background: #353b45; padding: 2px 5px; border-radius: 3px; font-family: monospace; color: #e5c07b;">\1</code>',
                text,
            )

            # Process formatting in list items
            text = bold_pattern.sub(r"<strong>\1</strong>", text)

            # Only escape parts of the text that aren't already HTML tags
            processed_text = ""
            current_pos = 0

            # Create a pattern to match all HTML tags we've inserted
            tag_pattern = re.compile(r"<(code|strong)[^>]*>.*?</(code|strong)>", re.DOTALL)
            for match in tag_pattern.finditer(text):
                # Escape text before the tag
                if current_pos < match.start():
                    processed_text += escape_html(text[current_pos : match.start()])
                # Add the tag unchanged
                processed_text += text[match.start() : match.end()]
                current_pos = match.end()

            # Escape any remaining text after the last tag
            if current_pos < len(text):
                processed_text += escape_html(text[current_pos:])

            # If no tags were found, escape the entire text
            if processed_text == "":
                processed_text = escape_html(text)

            processed_lines.append(
                f'<ul style="margin-left: {indent*10}px"><li>{processed_text}</li></ul>'
            )
            continue

        # Handle ordered lists with proper numbering
        ordered_list_match = re.match(r"^(\s*)(\d+)\.\s+(.+)$", line)
        if ordered_list_match:
            indent = len(ordered_list_match.group(1))
            number = ordered_list_match.group(2)  # Keep track of the actual number
            text = ordered_list_match.group(3)

            # Process inline code first - important to handle this before escaping HTML
            text = inline_code_pattern.sub(
                r'<code style="background: #353b45; padding: 2px 5px; border-radius: 3px; font-family: monospace; color: #e5c07b;">\1</code>',
                text,
            )

            # Process formatting in list items
            text = bold_pattern.sub(r"<strong>\1</strong>", text)

            # Only escape parts of the text that aren't already HTML tags
            processed_text = ""
            current_pos = 0

            # Create a pattern to match all HTML tags we've inserted
            tag_pattern = re.compile(r"<(code|strong)[^>]*>.*?</(code|strong)>", re.DOTALL)
            for match in tag_pattern.finditer(text):
                # Escape text before the tag
                if current_pos < match.start():
                    processed_text += escape_html(text[current_pos : match.start()])
                # Add the tag unchanged
                processed_text += text[match.start() : match.end()]
                current_pos = match.end()

            # Escape any remaining text after the last tag
            if current_pos < len(text):
                processed_text += escape_html(text[current_pos:])

            # If no tags were found, escape the entire text
            if processed_text == "":
                processed_text = escape_html(text)

            # Use the actual number from the markdown in the HTML
            processed_lines.append(
                f'<ol start="{number}" style="margin-left: {indent*10}px"><li>{processed_text}</li></ol>'
            )
            continue

        # Process bold and italic
        line = bold_pattern.sub(r"<strong>\1</strong>", line)
        line = italic_pattern.sub(r"<em>\1</em>", line)
        line = inline_code_pattern.sub(
            r'<code style="background: #353b45; padding: 2px 5px; border-radius: 3px; font-family: monospace; color: #e5c07b;">\1</code>',
            line,
        )

        # Process links - improved regex to better handle the closing tag
        line = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" style="color: #61afef;">\1</a>', line
        )

        # Process images
        line = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            r'<img src="\2" alt="\1" style="max-width: 100%; margin: 10px 0;">',
            line,
        )

        # Process strikethrough
        line = re.sub(r"~~([^~]+)~~", r"<del>\1</del>", line)

        # If it's not a special line, just add it with formatting already processed
        if line.strip():
            # Apply the same tag-preserving logic we used for lists
            processed_line = ""
            current_pos = 0

            # Create a pattern to match all HTML tags we've inserted
            tag_pattern = re.compile(
                r"<(code|strong|em|a|img|del)[^>]*>.*?</(code|strong|em|a|del)>|<img[^>]*>",
                re.DOTALL,
            )
            for match in tag_pattern.finditer(line):
                # Escape text before the tag
                if current_pos < match.start():
                    processed_line += escape_html(line[current_pos : match.start()])
                # Add the tag unchanged
                processed_line += line[match.start() : match.end()]
                current_pos = match.end()

            # Escape any remaining text after the last tag
            if current_pos < len(line):
                processed_line += escape_html(line[current_pos:])

            # If no tags were found, escape the entire text
            if processed_line == "":
                processed_line = escape_html(line)

            processed_lines.append(f"<p>{processed_line}</p>")
        else:
            processed_lines.append("<br>")

    html_content = "\n".join(processed_lines)

    # Debug any remaining placeholders before replacement
    for i in range(len(code_blocks)):
        placeholder = placeholder_pattern.format(i)
        if placeholder in html_content:
            logger.debug(f"Found placeholder {i} in html_content, will replace it")
        else:
            logger.debug(f"Placeholder {i} not found in html_content!")

    # Re-insert code blocks
    for i, (lang, code) in enumerate(code_blocks):
        placeholder = placeholder_pattern.format(i)
        escaped_code = escape_html(code)

        # Create a better styled code block
        html_code_block = """
        <div class="code-block">
            <div class="code-header">{}</div>
            <pre><code class="language-{}">{}</code></pre>
        </div>
        """.format(
            lang, lang, escaped_code
        )

        # Make sure all instances of the placeholder are replaced
        html_content = html_content.replace(placeholder, html_code_block)
    # do a final search for bad strongs which are #&lt;/strong&gt and replace them
    html_content = re.sub(r"&lt;/strong&gt;", "</strong>", html_content)
    # Style the HTML with CSS
    # Replace global styles with scoped styles by applying CSS classes with a unique prefix
    prefix = "md_display_"

    # Process HTML to add the scoped class to all elements
    html_with_scoped_classes = html_content

    # Add class attributes to common HTML elements
    for tag in [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "blockquote",
        "code",
        "pre",
        "a",
        "ul",
        "ol",
        "li",
        "hr",
        "strong",
        "em",
        "del",
        "img",
    ]:
        # Add class to opening tags that don't already have a class
        html_with_scoped_classes = re.sub(
            rf"<{tag}(?!\s+class=)(\s+[^>]*)?>",
            f'<{tag} class="{prefix}{tag}"\\1>',
            html_with_scoped_classes,
        )

    # Special handling for div.code-block and div.code-header
    html_with_scoped_classes = html_with_scoped_classes.replace(
        'class="code-block"', f'class="{prefix}code-block"'
    )
    html_with_scoped_classes = html_with_scoped_classes.replace(
        'class="code-header"', f'class="{prefix}code-header"'
    )

    styled_html = f"""
    <div class="{prefix}container" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                line-height: 1.5; 
                color: #f0f0f0; 
                overflow-y: auto;
                padding: 15px;
                font-size: 14px;
                background: #282c34;
                border-radius: 4px;">
        <style>
            /* Scoped styles using class prefix to prevent leaking */
            .{prefix}container .{prefix}h1, 
            .{prefix}container .{prefix}h2, 
            .{prefix}container .{prefix}h3, 
            .{prefix}container .{prefix}h4, 
            .{prefix}container .{prefix}h5, 
            .{prefix}container .{prefix}h6 {{ 
                color: #e6e6e6; 
                margin-top: 20px;
                margin-bottom: 10px;
                border-bottom: 1px solid #444;
                padding-bottom: 5px;
            }}
            .{prefix}container .{prefix}h1 {{ font-size: 24px; }}
            .{prefix}container .{prefix}h2 {{ font-size: 20px; }}
            .{prefix}container .{prefix}h3 {{ font-size: 18px; }}
            .{prefix}container .{prefix}h4 {{ font-size: 16px; }}
            .{prefix}container .{prefix}h5 {{ font-size: 14px; }}
            .{prefix}container .{prefix}h6 {{ font-size: 13px; }}
            .{prefix}container .{prefix}p {{ margin: 10px 0; }}
            .{prefix}container .{prefix}blockquote {{ 
                padding: 10px; 
                background: #353b45; 
                border-left: 4px solid #528bff; 
                margin: 10px 0;
            }}
            .{prefix}container .{prefix}code {{ 
                background: #353b45; 
                padding: 2px 5px; 
                border-radius: 3px; 
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                color: #e5c07b;
            }}
            .{prefix}container .{prefix}pre {{ 
                background: #2d333b; 
                padding: 10px; 
                border-radius: 3px; 
                overflow-x: auto; 
                margin: 10px 0;
                border: 1px solid #444;
            }}
            .{prefix}container .{prefix}pre .{prefix}code {{ 
                background: transparent; 
                padding: 0; 
                color: #abb2bf;
                display: block;
                line-height: 1.5;
            }}
            .{prefix}container .{prefix}code-block {{
                margin: 15px 0;
                border: 1px solid #444;
                border-radius: 5px;
                overflow: hidden;
            }}
            .{prefix}container .{prefix}code-header {{
                background: #2d333b;
                color: #abb2bf;
                padding: 5px 10px;
                font-family: monospace;
                font-size: 12px;
                border-bottom: 1px solid #444;
            }}
            .{prefix}container .{prefix}code-block .{prefix}pre {{
                margin: 0;
                border: none;
                border-radius: 0;
                max-height: 400px;
                overflow-y: auto;
            }}
            .{prefix}container .{prefix}a {{ color: #61afef; text-decoration: none; }}
            .{prefix}container .{prefix}a:hover {{ text-decoration: underline; }}
            .{prefix}container .{prefix}ul, .{prefix}container .{prefix}ol {{ margin: 10px 0; padding-left: 20px; }}
            .{prefix}container .{prefix}li {{ margin: 5px 0; }}
            .{prefix}container .{prefix}hr {{ border: none; border-top: 1px solid #444; margin: 20px 0; }}
            .{prefix}container .{prefix}strong {{ font-weight: bold; color: #d7d7d7; }}
            .{prefix}container .{prefix}em {{ font-style: italic; color: #c678dd; }}
            .{prefix}container .{prefix}del {{ text-decoration: line-through; color: #e06c75; }}
            .{prefix}container .{prefix}img {{ max-width: 100%; border-radius: 5px; }}
        </style>
        {html_with_scoped_classes}
    </div>
    """

    # Final check for any remaining placeholders
    placeholder_check = re.search(r"CODEBLOCKPLACEHOLDER\d+CODEBLOCK_", styled_html)
    if placeholder_check:
        logger.warning(
            f"Warning: Some code block placeholders were not replaced: {placeholder_check.group(0)}"
        )
        # Emergency fix - replace any remaining placeholders with a warning message
        styled_html = re.sub(
            r"CODEBLOCKPLACEHOLDER\d+CODEBLOCK_",
            f'<div class="{prefix}code-block"><div class="{prefix}code-header">ERROR</div><pre class="{prefix}pre"><code class="{prefix}code">Code block could not be properly rendered</code></pre></div>',
            styled_html,
        )

    return styled_html
