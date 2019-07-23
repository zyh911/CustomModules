import os
import fire

from azure.storage.blob import BlockBlobService


def import_data_from_blob(account_name, account_key, container, data_folder, output_folder):
    blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
    if len(data_folder) == 0 or data_folder[-1] != '/':
        data_folder += '/'
    prefix_len = len(data_folder)
    files = []
    for blob_path in blob_service.list_blob_names(container, prefix=data_folder):
        file_name = blob_path[prefix_len:]

        folder = os.path.dirname(os.path.join(output_folder, file_name))
        os.makedirs(folder, exist_ok=True)

        print(f"Start downloading file: {file_name}")
        blob_service.get_blob_to_path(
            container_name=container,
            blob_name=blob_path,
            file_path=os.path.join(output_folder, file_name),
        )
        print(f"End downloading file: {file_name}")
        files.append(file_name)
    print(f"{len(files)} files downloaded")


if __name__ == '__main__':
    fire.Fire(import_data_from_blob)
