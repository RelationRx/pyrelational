import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, cast

import requests


def fetch_data(api_url: str) -> Any:
    """Fetch data through an api.

    :param api_url: URL to api
    :return: response returned by api, assuming json format
    """
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()  # Assuming the data is in JSON format
    else:
        raise requests.RequestException(f"Failed to fetch data:{response.status_code}")


def download_file(url: str, directory: str = ".", filename: Optional[str] = None, num_chunks: int = 4) -> Optional[str]:
    """Downloads a file from a URL into a specified directory with parallel downloads."""
    if filename is None:
        filename = url.split("/")[-1]
    file_path = os.path.join(directory, filename)

    create_directory_if_not_exists(directory)
    total_size = get_file_size(url)

    chunk_size = total_size // num_chunks
    chunks: List[Optional[bytes]] = [None] * num_chunks

    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = [
            executor.submit(
                download_chunk,
                url,
                i * chunk_size,
                ((i + 1) * chunk_size) - 1 if i != num_chunks - 1 else total_size - 1,
                chunks,
                i,
            )
            for i in range(num_chunks)
        ]
        for future in futures:
            future.result()  # Ensure all futures complete

    chunks = validate_and_convert_chunks(chunks)
    combine_chunks_into_file(chunks, file_path)

    return file_path


def create_directory_if_not_exists(directory: str) -> None:
    """Ensure the directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            raise OSError(f"Error: Unable to create directory {directory}. {e}")


def get_file_size(url: str) -> int:
    """Retrieve the total size of the file from the URL."""
    try:
        with requests.head(url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            if total_size == 0:
                raise ValueError("Cannot download a file with unknown size.")
            return total_size
    except requests.RequestException as e:
        raise ValueError(f"Error retrieving file info: {e}")


def download_chunk(url: str, start: int, end: int, chunks: List[Optional[bytes]], index: int) -> None:
    """Download a specific range of bytes from a URL and store it in a list at the given index."""
    headers = {"Range": f"bytes={start}-{end}"}
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        chunks[index] = response.content
    except requests.RequestException as e:
        print(f"Error downloading chunk {index}: {e}")
        chunks[index] = None


def validate_and_convert_chunks(chunks: List[Optional[bytes]]) -> List[bytes]:
    """
    Validates that all chunks are successfully downloaded and converts the list of Optional[bytes]
    to a list of bytes. Raises an exception if any chunk is None.

    :param chunks: list containing downloaded file chunks, which may include None.
    :return: a list containing only bytes objects.
    """
    if any(chunk is None for chunk in chunks):
        raise ValueError("One or more file chunks could not be downloaded.")

    # Filter out None values and return a list of bytes (none of the chunks are None at this point)
    return [chunk for chunk in chunks if chunk is not None]


def combine_chunks_into_file(chunks: List[bytes], file_path: str) -> None:
    """Combine downloaded chunks and write them to the final file path."""
    try:
        with open(file_path, "wb") as f:
            for chunk in chunks:
                if chunk:
                    f.write(chunk)
    except IOError as e:
        raise IOError(f"Error writing file: {e}")
