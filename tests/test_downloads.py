import unittest
from unittest.mock import Mock, call

from requests import Response
from requests.exceptions import ChunkedEncodingError, HTTPError

from scripts.model_downloads import download_snapshot


class DownloadTests(unittest.TestCase):
    def test_interrupted_stream_is_retried_without_discarding_cache(self):
        download = Mock(side_effect=[ChunkedEncodingError("IncompleteRead"), "/cache/snapshot"])
        pause = Mock()
        self.assertEqual(download_snapshot("model", "pinned-sha", download=download, pause=pause), "/cache/snapshot")
        self.assertEqual(download.call_count, 2)
        first = download.call_args_list[0].kwargs
        self.assertEqual(first, download.call_args_list[1].kwargs)
        self.assertEqual(first["max_workers"], 1)
        self.assertEqual(first["revision"], "pinned-sha")
        self.assertNotIn("force_download", first)
        pause.assert_called_once_with(5)

    def test_retry_limit(self):
        download = Mock(side_effect=ChunkedEncodingError("IncompleteRead"))
        pause = Mock()
        with self.assertRaises(ChunkedEncodingError):
            download_snapshot("model", "sha", download=download, pause=pause, attempts=3)
        self.assertEqual(download.call_count, 3)
        self.assertEqual(pause.call_args_list, [call(5), call(10)])

    def test_disk_error_is_not_retried(self):
        download = Mock(side_effect=OSError("No space left on device"))
        pause = Mock()
        with self.assertRaises(OSError):
            download_snapshot("model", "sha", download=download, pause=pause)
        self.assertEqual(download.call_count, 1)
        pause.assert_not_called()

    def test_permission_denial_is_not_retried(self):
        response = Response()
        response.status_code = 403
        download = Mock(side_effect=HTTPError("Forbidden", response=response))
        pause = Mock()
        with self.assertRaises(HTTPError):
            download_snapshot("model", "sha", download=download, pause=pause)
        self.assertEqual(download.call_count, 1)
        pause.assert_not_called()

    def test_http_503_is_retried(self):
        response = Response()
        response.status_code = 503
        download = Mock(side_effect=[HTTPError("Unavailable", response=response), "/cache/snapshot"])
        self.assertEqual(download_snapshot("model", "sha", download=download, pause=Mock()), "/cache/snapshot")
