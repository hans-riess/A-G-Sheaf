import numpy as np
import os
import shutil
import subprocess
import yaml
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from pathlib import Path
from typing import Optional, Union

from matplotlib.figure import Figure


def load_experiment_config(
    config_path: Union[str, Path],
    defaults_path: Union[str, Path]
) -> dict:
    """
    Loads an experiment configuration on top of the general defaults.

    The defaults hold everything that is not specific to one experiment -- colors, video
    settings, tile costs and scores -- and the experiment file holds the world being run
    and anything it wants to depart from. The two are merged section by section, so an
    experiment overriding one tile cost keeps the rest rather than replacing the block.

    A key given in the experiment file always wins, including when it is left blank: a
    bare "seed:" is an explicit request for a random seed, not a missing entry.

    Args:
        config_path (str | Path): Path to the experiment's configuration file.
        defaults_path (str | Path): Path to the general defaults file.
    """
    with open(defaults_path) as defaults_file:
        defaults = yaml.safe_load(defaults_file) or {}
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file) or {}

    assert isinstance(defaults, dict), "Failed to load defaults from %s. Expected a mapping, received type %r." % (defaults_path, type(defaults).__name__)
    assert isinstance(config, dict), "Failed to load configuration from %s. Expected a mapping, received type %r." % (config_path, type(config).__name__)

    return _merge_settings(defaults, config)


def run_output_path(
    directory: Union[str, Path],
    timestamp: str,
    extension: str,
    number: Optional[int] = None
) -> Path:
    """
    Names one of a run's output files. The timestamp names the run and nothing about what was
    configured goes in the name, so the file describes itself through the log rather than
    through what it is called.

    A run producing one file of a kind gets "<timestamp>.<extension>". A run producing several
    videos, one per view, numbers them "<timestamp>-1.mp4", "<timestamp>-2.mp4" and so on.

    Args:
        directory (str | Path): The directory the file belongs in.
        timestamp (str): The timestamp naming this run.
        extension (str): The file extension, without a dot.
        number (Optional[int]): Which of several files this is, counting from 1, or None when
            the run produces only one of them.
    """
    stem = timestamp if number is None else "%s-%d" % (timestamp, number)
    return Path(directory) / ("%s.%s" % (stem, extension))


def _merge_settings(defaults: dict, overrides: dict) -> dict:
    """
    Merges one mapping of settings over another, section by section. Nested mappings are
    merged, so a partial section overrides only the keys it names; anything else, lists
    included, is replaced outright.

    Args:
        defaults (dict): The settings being overridden.
        overrides (dict): The settings taking precedence.
    """
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged


def compose_config(
    experiment: Union[str, Path],
    layers: Optional[list] = None,
    defaults_path: Optional[Union[str, Path]] = None,
    announce: bool = False
) -> dict:
    """
    The configuration a run is run from: the defaults, then the experiment file, then each
    layer in turn, merged section by section.

    This is the layering exp/robotarium/build_submission.py applies when it composes a testbed
    configuration out of `--config a b c`, kept in one place so that the runner, the frame
    preview and the submission builder compose a configuration the same way rather than from
    separate copies that could drift apart.

    Args:
        experiment (str | Path): The experiment file, applied over the defaults.
        layers (Optional[list]): Further layers, in order, each applied over the ones before it.
        defaults_path (Optional[str | Path]): The general defaults. Defaults to exp/defaults.yaml
            beside this repository's experiments.
        announce (bool): Whether to print each layer as it is applied.
    """
    if defaults_path is None:
        defaults_path = Path(__file__).resolve().parents[2] / "exp" / "defaults.yaml"

    return merge_layers(load_experiment_config(experiment, defaults_path), layers, announce)


def merge_layers(config: dict, layers: Optional[list] = None, announce: bool = False) -> dict:
    """
    Applies further configuration layers over an already-merged configuration, in order.

    Split out from compose_config so that a world read from a file and a world generated in
    memory take layers by the same code path.

    Args:
        config (dict): The configuration being layered over.
        layers (Optional[list]): Paths to the layers, in order, each applied over the ones before.
        announce (bool): Whether to print each layer as it is applied.
    """
    for layer in (layers or []):
        layer = Path(layer)
        assert layer.is_file(), \
            "Failed to compose the configuration. No layer at %s." % layer
        overrides = yaml.safe_load(layer.read_text()) or {}
        assert isinstance(overrides, dict), \
            "Failed to compose the configuration. Expected a mapping in %s, received type %r." % (
                layer, type(overrides).__name__)
        config = _merge_settings(config, overrides)
        if announce:
            print("Layer: %s" % layer)

    return config


def apply_overrides(config: dict, assignments: Optional[list] = None) -> dict:
    """
    Applies `dotted.key=value` overrides over an already-merged configuration, returning a new
    one. This is the command line's equivalent of writing a one-key layer file.

    Values are parsed as YAML rather than kept as strings, so that `false`, `8` and `0.05`
    arrive as the bool, int and float they would have been had they been written into the
    configuration file, and quoting still forces a string where one is wanted.

    Only keys the configuration already holds may be set. A run silently shaped by a typo --
    `sheaf.enable=false` leaving the flow on -- is the failure this rules out, and the price is
    that a genuinely new key has to be introduced in a file rather than from the command line.

    Args:
        config (dict): The merged configuration to override.
        assignments (Optional[list]): Strings of the form "dotted.key=value".
    """
    overridden = dict(config)

    for assignment in (assignments or []):
        assert "=" in assignment, \
            "Failed to apply the override %r. Expected the form dotted.key=value." % assignment
        path, _, raw = assignment.partition("=")
        keys = [key for key in path.strip().split(".") if key]
        assert keys, \
            "Failed to apply the override %r. It names no key." % assignment

        section = overridden
        for depth, key in enumerate(keys[:-1]):
            assert isinstance(section.get(key), dict), \
                "Failed to apply the override %r. %r is not a section of the configuration." % (
                    assignment, ".".join(keys[:depth + 1]))
            section[key] = dict(section[key])
            section = section[key]

        assert keys[-1] in section, \
            "Failed to apply the override %r. The configuration has no key %r; a key that does " \
            "not already exist has to be introduced in a configuration file, so that a typo " \
            "here fails rather than quietly doing nothing." % (assignment, path.strip())

        section[keys[-1]] = yaml.safe_load(raw)

    return overridden


class VideoSaver:
    """
    Class to save frames from a Matplotlib figure to an MP4 file.

    Frames are piped to an `ffmpeg` subprocess encoding H.264/yuv420p with
    a moved moov atom (`+faststart`) rather than through cv2.VideoWriter's
    "mp4v" fourcc, which writes MPEG-4 Part 2 -- a codec no major browser
    decodes, so those videos open in a media player but not inline (VS
    Code's Simple Browser included). cv2 is otherwise unused in this
    project, so this drops that dependency for video entirely.
    """

    def __init__(
        self,
        figure: Figure,
        filename: str,
        fps: int = 30
    ):
        """
        Initialize the video saver.

        Args:
            figure (Figure): The Matplotlib figure to record.
            filename (str): Output path for the video file (e.g. "video.mp4").
            fps (int): Frames per second for the output video.
        """
        self.figure = figure
        self.filename = filename
        self.fps = fps
        self.process = None
        self.closed = False

    def writeFrame(self):
        """
        Capture the current figure state and append it as a frame.
        """
        if self.closed:
            raise RuntimeError("Cannot write frame: VideoSaver is already closed.")

        # Force canvas draw
        self.figure.canvas.draw()

        # Pull the frame
        buf = np.asarray(self.figure.canvas.buffer_rgba())
        frame_RGB = np.ascontiguousarray(buf[:, :, :3])

        # Init the encoder process if needed
        if self.process is None:
            height, width = frame_RGB.shape[:2]
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin is None:
                raise RuntimeError(
                    "ffmpeg not found on PATH. VideoSaver pipes frames to it to encode "
                    "web-playable (H.264) video; install ffmpeg to record videos.")
            # libx264 requires even dimensions under yuv420p's 2x2 chroma subsampling;
            # the crop scale filter rounds a figure of odd pixel size down to fit.
            command = [
                ffmpeg_bin, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", "%dx%d" % (width, height), "-r", str(self.fps),
                "-i", "-",
                "-an",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(self.filename),
            ]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
            if self.process.stdin is None:
                raise RuntimeError("Failed to open a stdin pipe to ffmpeg for %s." % self.filename)

        # Write frame
        self.process.stdin.write(frame_RGB.tobytes())

    def close(self):
        """
        Finalize and close the video file.
        """
        if not self.closed:
            if self.process is not None:
                self.process.stdin.close()
                return_code = self.process.wait()
                if return_code != 0:
                    raise RuntimeError(
                        "ffmpeg exited with code %d while writing %s." % (return_code, self.filename))
            self.closed = True