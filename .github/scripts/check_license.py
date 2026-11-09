"""Reject an absent or unfinished license before a release build."""

from pathlib import Path
import sys


def main():
    text = Path('LICENSE').read_text(encoding='utf-8')
    if '[year]' in text or '[fullname]' in text:
        raise ValueError('LICENSE still contains copyright placeholders')
    print('LICENSE exists and has no copyright placeholders')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f'License check failed: {error}')
