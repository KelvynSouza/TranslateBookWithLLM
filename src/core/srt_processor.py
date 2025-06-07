import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SRTProcessor:
    def __init__(self):
        self.subtitle_pattern = re.compile(
            r'(\d+)\s*\n'
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n'
            r'((?:(?!\n\n|\n\d+\s*\n).*\n?)*)',
            re.MULTILINE
        )

    def parse_srt(self, content: str) -> List[Dict[str, str]]:
        subtitles = []

        content = content.replace('\r\n', '\n').replace('\r', '\n')
        if not content.endswith('\n'):
            content += '\n'

        blocks = content.split('\n\n')

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split('\n')
            if len(lines) < 3:
                continue

            if not lines[0].isdigit():
                continue
            number = lines[0]

            timecode_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
            if not timecode_match:
                continue
            start_time, end_time = timecode_match.groups()

            text_lines = lines[2:]
            text = '\n'.join(text_lines)

            subtitle = {
                'number': number,
                'start_time': start_time,
                'end_time': end_time,
                'text': text,
                'original_text': text
            }
            subtitles.append(subtitle)

        logger.info(f"Parsed {len(subtitles)} subtitles from SRT file")
        return subtitles

    def extract_translatable_text(self, subtitles: List[Dict[str, str]]) -> List[Tuple[int, str]]:
        translatable = []

        for idx, subtitle in enumerate(subtitles):
            if subtitle['text'].strip():
                translatable.append((idx, subtitle['text']))

        return translatable

    def update_translated_subtitles(self, subtitles: List[Dict[str, str]],
                                    translations: Dict[int, str]) -> List[Dict[str, str]]:
        for idx, translation in translations.items():
            if idx < len(subtitles):
                subtitles[idx]['text'] = translation

        return subtitles

    def reconstruct_srt(self, subtitles: List[Dict[str, str]]) -> str:
        srt_content = []

        for subtitle in subtitles:
            block = f"{subtitle['number']}\n"
            block += f"{subtitle['start_time']} --> {subtitle['end_time']}\n"
            block += f"{subtitle['text']}\n"

            srt_content.append(block)

        return '\n'.join(srt_content)

    def validate_srt(self, content: str) -> bool:
        return bool(self.subtitle_pattern.search(content))

    def merge_multiline_subtitles(self, subtitles: List[Dict[str, str]],
                                  max_chars: int = 100) -> List[Dict[str, str]]:
        if not subtitles:
            return subtitles

        merged = []
        current = None

        for subtitle in subtitles:
            if current is None:
                current = subtitle.copy()
            elif (len(current['text']) + len(subtitle['text']) + 1 <= max_chars and
                  self._is_continuation(current['text'], subtitle['text'])):
                current['text'] += ' ' + subtitle['text']
                current['end_time'] = subtitle['end_time']
                if 'merged_numbers' not in current:
                    current['merged_numbers'] = [current['number']]
                current['merged_numbers'].append(subtitle['number'])
            else:
                merged.append(current)
                current = subtitle.copy()

        if current:
            merged.append(current)

        return merged

    def _is_continuation(self, text1: str, text2: str) -> bool:
        terminators = '.!?'

        if not text1 or not text2:
            return False

        ends_with_terminator = text1.rstrip()[-1] in terminators if text1.rstrip() else False
        starts_with_capital = text2.strip()[0].isupper() if text2.strip() else False
        starts_with_i = text2.strip().startswith('I ') or text2.strip() == 'I'

        return not ends_with_terminator and (not starts_with_capital or starts_with_i)

    def group_subtitles_for_translation(self, subtitles: List[Dict[str, str]],
                                        lines_per_block: int = 5,
                                        max_chars_per_block: int = 500) -> List[List[Dict[str, str]]]:
        if not subtitles:
            return []

        blocks = []
        current_block = []
        current_char_count = 0

        for i, subtitle in enumerate(subtitles):
            text = subtitle.get('text', '').strip()

            if not text:
                if current_block:
                    current_block.append(subtitle)
                continue

            text_length = len(text)

            would_exceed_lines = len(current_block) >= lines_per_block
            would_exceed_chars = current_char_count + text_length > max_chars_per_block

            if current_block and (would_exceed_lines or would_exceed_chars):
                blocks.append(current_block)
                current_block = []
                current_char_count = 0

            current_block.append(subtitle)
            current_char_count += text_length

        if current_block:
            blocks.append(current_block)

        logger.info(f"Grouped {len(subtitles)} subtitles into {len(blocks)} blocks")
        return blocks

    def extract_block_translations(self, translated_text: str, block_indices: List[int]) -> Dict[int, str]:
        translations = {}

        preprocessed_text = self._fix_multiple_indices_on_same_line(translated_text)
        preprocessed_text = self._fix_missing_indices(preprocessed_text, block_indices)

        lines = preprocessed_text.strip().split('\n')
        current_index = None
        current_text_lines = []

        for line in lines:
            index_match = re.match(r'^\[(\d+)\](.*)$', line)

            if index_match:
                if current_index is not None and current_text_lines:
                    translations[current_index] = '\n'.join(current_text_lines).strip()

                current_index = int(index_match.group(1))
                remaining_text = index_match.group(2).strip()

                if remaining_text:
                    current_text_lines = [remaining_text]
                else:
                    current_text_lines = []
            else:
                if current_index is not None:
                    current_text_lines.append(line)

        if current_index is not None and current_text_lines:
            translations[current_index] = '\n'.join(current_text_lines).strip()

        missing_indices = set(block_indices) - set(translations.keys())
        if missing_indices:
            logger.warning(f"Missing translations for indices: {missing_indices}")

        return translations

    def _fix_multiple_indices_on_same_line(self, text: str) -> str:
        pattern = r'(\S.*?)\s+(\[\d+\])'
        result = re.sub(pattern, r'\1\n\2', text)

        if result != text:
            logger.info(f"Fixed multiple indices on same line - separated indices onto new lines")

        return result

    def _fix_missing_indices(self, text: str, expected_indices: List[int]) -> str:
        lines = text.strip().split('\n')

        present_indices = set()
        for line in lines:
            match = re.match(r'^\[(\d+)\]', line)
            if match:
                present_indices.add(int(match.group(1)))

        missing_indices = set(expected_indices) - present_indices

        if missing_indices:
            logger.warning(f"LLM forgot to include indices: {sorted(missing_indices)}")

        return text
