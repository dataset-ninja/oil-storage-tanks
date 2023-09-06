# https://www.kaggle.com/datasets/towardsentropy/oil-storage-tanks
#

import csv
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s
from dataset_tools.convert import unpack_if_archive


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(desc=f"Downloading '{file_name_with_ext}' to buffer...", total=fsize) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "Oil Storage Tanks"
    dataset_path = "/mnt/d/datasetninja-raw/oil-storage-tanks/Oil Tanks"
    batch_size = 5

    bboxes_path = os.path.join(dataset_path, "labels_coco.json")
    big_files_data = os.path.join(dataset_path, "large_image_data.csv")

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        image_name = get_file_name_with_ext(image_path)
        image_bboxes = image_name_to_ann_data.get(image_name)

        if image_bboxes is not None:
            for box in image_bboxes:
                left = box[0]
                right = box[0] + box[2]
                top = box[1]
                bottom = box[1] + box[3]
                rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                label = sly.Label(rectangle, obj_class)
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    def create_annfor_big(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        image_name = get_file_name_with_ext(image_path)
        image_tags = image_name_to_tags.get(image_name)
        if image_tags is not None:
            latitude = sly.Tag(tag_latitude, value=image_tags[0])
            longiude = sly.Tag(tag_longiude, value=image_tags[1])
            location = sly.Tag(tag_location, value=image_tags[2])

        return sly.Annotation(
            img_size=(img_height, img_wight), labels=labels, img_tags=[latitude, longiude, location]
        )

    obj_class = sly.ObjClass("floating head tank", sly.Rectangle)

    tag_latitude = sly.TagMeta("latitude", sly.TagValueType.ANY_STRING)
    tag_longiude = sly.TagMeta("longiude", sly.TagValueType.ANY_STRING)
    tag_location = sly.TagMeta("location", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class], tag_metas=[tag_latitude, tag_longiude, tag_location]
    )
    api.project.update_meta(project.id, meta.to_json())

    image_name_to_tags = {}
    with open(big_files_data, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            image_name_to_tags[row[0]] = row[2:]

    image_id_to_name = {}
    image_name_to_ann_data = defaultdict(list)

    bboxes_data = load_json_file(bboxes_path)

    for curr_image_info in bboxes_data["images"]:
        image_id_to_name[curr_image_info["id"]] = curr_image_info["file_name"]

    for curr_ann_data in bboxes_data["annotations"]:
        image_id = curr_ann_data["image_id"]
        image_name_to_ann_data[image_id_to_name[image_id]].append(curr_ann_data["bbox"])

    for ds_name in os.listdir(dataset_path):
        images_path = os.path.join(dataset_path, ds_name)
        if dir_exists(images_path):
            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

            images_path = os.path.join(dataset_path, ds_name)
            images_names = os.listdir(images_path)

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                images_pathes_batch = [
                    os.path.join(images_path, image_name) for image_name in img_names_batch
                ]

                img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                if ds_name == "image_patches":
                    anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
                else:
                    anns_batch = [
                        create_annfor_big(image_path) for image_path in images_pathes_batch
                    ]
                api.annotation.upload_anns(img_ids, anns_batch)

                progress.iters_done_report(len(img_names_batch))
    return project
