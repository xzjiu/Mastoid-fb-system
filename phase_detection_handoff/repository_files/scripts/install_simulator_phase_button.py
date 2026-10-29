"""Add a phase-page launcher to the inspected PyQt5 study GUI."""
import argparse
import ast
from datetime import datetime
from pathlib import Path

MARKER = '# Local phase detection launcher'
ANCHOR = '        self.show()'
BLOCK = '''        # Local phase detection launcher
        self.button_phase_detection = QtWidgets.QPushButton('Phase Detection', self)
        self.button_phase_detection.setObjectName('button_phase_detection')
        self.button_phase_detection.setToolTip(
            'Open the local phase detection page for a completed recording')
        self.volumes_grid.addWidget(
            self.button_phase_detection, self.volumes_grid.rowCount(), 0, 1, 3)

        def open_phase_detection():
            if self._recording_study:
                QtWidgets.QMessageBox.information(
                    self, 'Phase Detection',
                    'Stop recording and wait for the recording to finish saving first.')
                return
            from PyQt5.QtCore import QUrl
            from PyQt5.QtGui import QDesktopServices
            if not QDesktopServices.openUrl(QUrl('http://127.0.0.1:8795/')):
                QtWidgets.QMessageBox.information(
                    self, 'Phase Detection',
                    'Open http://127.0.0.1:8795/ in your browser. '
                    'The phase runner must be running on this computer.')

        self.button_phase_detection.clicked.connect(open_phase_detection)

'''


def patched(source):
    ast.parse(source)
    if MARKER in source:
        return source
    if source.count(ANCHOR) != 1:
        raise ValueError('Unexpected GUI version: expected one self.show() call. No files changed.')
    for required in ('self.volumes_grid', 'self._recording_study', 'class Ui(QtWidgets.QWidget)'):
        if required not in source:
            raise ValueError('Unsupported GUI source. No files changed.')
    result = source.replace(ANCHOR, BLOCK + ANCHOR, 1)
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui-dir', required=True, type=Path)
    args = parser.parse_args()
    path = args.gui_dir.expanduser().resolve() / 'study_gui.py'
    original_bytes = path.read_bytes()
    source = original_bytes.decode('utf-8')
    result = patched(source)
    if result == source:
        print('Phase Detection button already installed.')
        return
    backup = path.with_name('study_gui.py.before_phase_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.write_bytes(original_bytes)
    path.write_text(result, encoding='utf-8')
    print('Added Phase Detection button:', path)
    print('Backup:', backup)
    print('Start the phase runner, then launch the study GUI using its normal environment.')
    print('The button opens http://127.0.0.1:8795/; it does not auto-select a session or start inference.')


if __name__ == '__main__':
    main()
