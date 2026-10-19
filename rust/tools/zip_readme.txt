dpmread - Data Matrix (ECC 200) reader for poor quality marks

  dpmread-gui    drop images or folders on the window
  dpmread        command line: dpmread IMAGE_OR_FOLDER [...] [--json]
                 dpmread --help lists every option

Both programs include a trained symbol detector and a module classifier;
turn them off in the GUI settings (Trained models) or with --no-detector
and --no-classifier.

Nothing to install: unpack and run. On Linux the libxkbcommon and
libxcb-xkb files next to dpmread-gui are found by it automatically; keep
them in the same folder.

Documentation, source and issues: https://github.com/aharjajarvi/dpmread

Licence: Apache License 2.0 (LICENSE). Ported code and shipped libraries:
NOTICE. Licences of everything linked into the programs:
THIRD_PARTY_LICENSES.md. This software is based in part on the work of
the Independent JPEG Group.
