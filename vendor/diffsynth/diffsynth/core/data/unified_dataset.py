from .operators import *
import os, torch, json, pandas, random, logging

logger = logging.getLogger(__name__)


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        cached_data_operator=None,
        special_operator_map=None,
        max_data_items=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle() if cached_data_operator is None else cached_data_operator
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.data = []
        self.cached_data = []
        self.train_indices = []
        self.val_indices = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
        self._compute_split_indices()
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)

    def load_cached_split_map(self):
        split_map = {}
        manifest_files = []
        if self.base_path is None or not os.path.isdir(self.base_path):
            return split_map, manifest_files

        for file_name in sorted(os.listdir(self.base_path)):
            if not (file_name.startswith("cache_manifest_rank") and file_name.endswith(".jsonl")):
                continue
            manifest_path = os.path.join(self.base_path, file_name)
            if not os.path.isfile(manifest_path):
                continue
            manifest_files.append(manifest_path)
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cache_path = payload.get("path")
                    if cache_path is None:
                        continue
                    cache_path = str(cache_path)
                    if os.path.isabs(cache_path):
                        resolved_path = os.path.normpath(cache_path)
                    else:
                        resolved_path = os.path.normpath(os.path.join(self.base_path, cache_path))
                    split_map[resolved_path] = str(payload.get("split", "train")).strip().lower()
        return split_map, manifest_files
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            self.cached_data = sorted(os.path.normpath(path) for path in self.cached_data)
            split_map, manifest_files = self.load_cached_split_map()
            self.data = [{"split": split_map.get(path, "train")} for path in self.cached_data]
            if manifest_files:
                print(f"Loaded cached split manifest from {len(manifest_files)} file(s).")
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def _compute_split_indices(self):
        """Compute train and val index lists based on the 'split' column in metadata."""
        self.train_indices = []
        self.val_indices = []
        for i, item in enumerate(self.data):
            split_value = item.get("split", "train") if isinstance(item, dict) else "train"
            split_value = str(split_value).strip().lower()
            if split_value in ("val", "test"):
                self.val_indices.append(i)
            else:
                self.train_indices.append(i)

    def __getitem__(self, data_id, _retry_count=0):
        max_retries = 10
        cache_path = None
        try:
            if self.load_from_cache:
                cache_path = self.cached_data[data_id % len(self.cached_data)]
                data = self.cached_data_operator(cache_path)
            else:
                data = self.data[data_id % len(self.data)].copy()
                for key in self.data_file_keys:
                    if key in data:
                        if key in self.special_operator_map:
                            data[key] = self.special_operator_map[key](data[key])
                        elif key in self.data_file_keys:
                            data[key] = self.main_data_operator(data[key])
            return data
        except Exception as e:
            source_detail = f"index {data_id}"
            if cache_path is not None:
                source_detail += f" (cache_path={cache_path})"
            if _retry_count >= max_retries:
                raise RuntimeError(
                    f"Failed to load data after {max_retries} retries. Last error at {source_detail}: {e}"
                ) from e
            logger.warning(
                f"Skipping data {source_detail} due to error ({type(e).__name__}: {e}). "
                f"Retry {_retry_count + 1}/{max_retries}."
            )
            new_id = random.randint(0, len(self) - 1)
            return self.__getitem__(new_id, _retry_count=_retry_count + 1)

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
