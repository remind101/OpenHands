import json
from unittest.mock import MagicMock, patch

import pytest

from openhands.core.config import AppConfig, SandboxConfig
from openhands.events.stream import EventStream
from openhands.runtime.impl.docker.docker_runtime import DockerRuntime, logger


# Basic AppConfig for testing
@pytest.fixture
def test_config():
    return AppConfig(
        workspace_base='/test/workspace',
        workspace_mount_path='/test/workspace',
        workspace_mount_path_in_sandbox='/workspace',
        sandbox=SandboxConfig(
            base_container_image='test_image',
            runtime_container_image='test_image',  # Assume image is pre-built for these tests
        ),
    )


@pytest.fixture
def mock_event_stream():
    return MagicMock(spec=EventStream)


# Mocks needed to instantiate DockerRuntime without actually interacting with Docker
@pytest.fixture(autouse=True)
def mock_docker_dependencies():
    with patch(
        'openhands.runtime.impl.docker.docker_runtime.docker.from_env'
    ) as mock_from_env, patch(
        'openhands.runtime.impl.docker.docker_runtime.build_runtime_image'
    ) as mock_build_image, patch(
        'openhands.runtime.impl.docker.docker_runtime.find_available_tcp_port',
        side_effect=[30000, 40000, 50000, 55000],
    ) as _, patch('openhands.runtime.impl.docker.docker_runtime.LogStreamer'), patch(
        'openhands.runtime.impl.docker.docker_runtime.add_shutdown_listener'
    ):
        mock_docker_client = MagicMock()
        mock_container = MagicMock()
        mock_docker_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_docker_client
        mock_build_image.return_value = 'built_test_image'  # Return a dummy image name
        yield mock_docker_client  # Yield the client mock for potential assertions


# =============================================
# Tests for docker_runtime_kwargs parsing
# =============================================


def test_kwargs_parsing_valid_json(
    test_config, mock_event_stream, mock_docker_dependencies
):
    """Test parsing of valid JSON string in docker_runtime_kwargs."""
    kwargs_dict = {'network_mode': 'host', 'labels': {'test': 'label'}}
    test_config.sandbox.docker_runtime_kwargs = json.dumps(kwargs_dict)

    runtime = DockerRuntime(test_config, mock_event_stream)
    runtime._init_container()  # Trigger the parsing and container run call

    # Assert that containers.run was called with the parsed kwargs
    mock_docker_dependencies.containers.run.assert_called_once()
    call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
    assert call_kwargs.get('network_mode') == 'host'
    assert call_kwargs.get('labels') == {'test': 'label'}
    assert 'volumes' not in call_kwargs  # Volumes are handled separately


def test_kwargs_parsing_invalid_json(
    test_config, mock_event_stream, mock_docker_dependencies
):
    """Test handling of invalid JSON string."""
    test_config.sandbox.docker_runtime_kwargs = '{"network_mode": "host", invalid_json'
    with patch.object(logger, 'error') as mock_log_error:
        runtime = DockerRuntime(test_config, mock_event_stream)
        runtime._init_container()

        mock_log_error.assert_called_with(
            'Failed to parse docker_runtime_kwargs JSON: Expecting property name enclosed in double quotes: line 1 column 26 (char 25)'
        )
        # Assert that containers.run was called without the invalid kwargs
        mock_docker_dependencies.containers.run.assert_called_once()
        call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
        assert 'network_mode' not in call_kwargs  # Kwarg should not be applied


def test_kwargs_parsing_non_dict_json(
    test_config, mock_event_stream, mock_docker_dependencies
):
    """Test handling of JSON string that is not a dictionary."""
    test_config.sandbox.docker_runtime_kwargs = json.dumps(['list', 'item'])
    with patch.object(logger, 'error') as mock_log_error:
        runtime = DockerRuntime(test_config, mock_event_stream)
        runtime._init_container()

        mock_log_error.assert_called_with(
            'Invalid docker_runtime_kwargs: docker_runtime_kwargs must be a JSON object'
        )
        # Assert that containers.run was called without the invalid kwargs
        mock_docker_dependencies.containers.run.assert_called_once()
        call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
        # Check a default or expected kwarg to ensure call happened, but not the invalid one
        assert call_kwargs.get('working_dir') == '/openhands/code/'


def test_kwargs_parsing_none(test_config, mock_event_stream, mock_docker_dependencies):
    """Test handling when docker_runtime_kwargs is None."""
    test_config.sandbox.docker_runtime_kwargs = None

    runtime = DockerRuntime(test_config, mock_event_stream)
    runtime._init_container()

    # Assert that containers.run was called without extra kwargs
    mock_docker_dependencies.containers.run.assert_called_once()
    call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
    # Check that only expected default/base kwargs are present
    expected_keys = {
        'image',
        'command',
        'entrypoint',
        'network_mode',
        'ports',
        'working_dir',
        'name',
        'detach',
        'environment',
        'volumes',
        'device_requests',
    }
    present_kwargs = {
        k for k, v in call_kwargs.items() if v is not None
    }  # Get keys of non-None values passed
    # Allow for potential None values like device_requests if GPU not enabled
    assert present_kwargs.issubset(expected_keys)


# =============================================
# Tests for volume merging logic
# =============================================


def test_volume_merging_workspace_only(
    test_config, mock_event_stream, mock_docker_dependencies
):
    """Test volume setup when only workspace mount is configured."""
    test_config.sandbox.docker_runtime_kwargs = None  # No custom volumes

    runtime = DockerRuntime(test_config, mock_event_stream)
    runtime._init_container()

    mock_docker_dependencies.containers.run.assert_called_once()
    call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
    expected_volumes = {
        test_config.workspace_mount_path: {
            'bind': test_config.workspace_mount_path_in_sandbox,
            'mode': 'rw',
        }
    }
    assert call_kwargs.get('volumes') == expected_volumes


def test_volume_merging_custom_only(
    test_config, mock_event_stream, mock_docker_dependencies
):
    """Test volume setup when only custom volumes are provided via kwargs."""
    # Disable workspace mount for this test
    test_config.workspace_mount_path = None
    test_config.workspace_mount_path_in_sandbox = None

    custom_volumes = {
        '/host/data': {'bind': '/data', 'mode': 'ro'},
        '/host/config': {'bind': '/config', 'mode': 'rw'},
    }
    test_config.sandbox.docker_runtime_kwargs = json.dumps({'volumes': custom_volumes})

    runtime = DockerRuntime(test_config, mock_event_stream)
    runtime._init_container()

    mock_docker_dependencies.containers.run.assert_called_once()
    call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
    assert call_kwargs.get('volumes') == custom_volumes
    assert (
        'volumes' not in call_kwargs
    )  # Ensure 'volumes' was removed from the kwargs passed via **


def test_volume_merging_workspace_and_custom(
    test_config, mock_event_stream, mock_docker_dependencies
):
    """Test merging workspace and custom volumes (no conflicts)."""
    custom_volumes = {
        '/host/data': {'bind': '/data', 'mode': 'ro'},
    }
    test_config.sandbox.docker_runtime_kwargs = json.dumps({'volumes': custom_volumes})

    runtime = DockerRuntime(test_config, mock_event_stream)
    runtime._init_container()

    mock_docker_dependencies.containers.run.assert_called_once()
    call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
    expected_volumes = {
        test_config.workspace_mount_path: {
            'bind': test_config.workspace_mount_path_in_sandbox,
            'mode': 'rw',
        },
        '/host/data': {'bind': '/data', 'mode': 'ro'},
    }
    assert call_kwargs.get('volumes') == expected_volumes
    assert 'volumes' not in call_kwargs


def test_volume_merging_conflict_workspace_priority(
    test_config, mock_event_stream, mock_docker_dependencies
):
    """Test conflict: custom volume targets same path as workspace. Workspace should win."""
    custom_volumes = {
        '/host/custom_workspace': {
            'bind': test_config.workspace_mount_path_in_sandbox,
            'mode': 'ro',
        },  # Conflict!
        '/host/data': {'bind': '/data', 'mode': 'rw'},
    }
    test_config.sandbox.docker_runtime_kwargs = json.dumps({'volumes': custom_volumes})

    with patch.object(logger, 'warning') as mock_log_warning:
        runtime = DockerRuntime(test_config, mock_event_stream)
        runtime._init_container()

        mock_log_warning.assert_any_call(
            f'Custom volume mount for {test_config.workspace_mount_path_in_sandbox} conflicts with workspace mount. Prioritizing workspace mount.'
        )

        mock_docker_dependencies.containers.run.assert_called_once()
        call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
        expected_volumes = {
            # Workspace mount should be present
            test_config.workspace_mount_path: {
                'bind': test_config.workspace_mount_path_in_sandbox,
                'mode': 'rw',
            },
            # Non-conflicting custom volume should be present
            '/host/data': {'bind': '/data', 'mode': 'rw'},
            # Conflicting custom volume should NOT be present
        }
        # Check that the conflicting host path is not in the final volumes
        assert '/host/custom_workspace' not in call_kwargs.get('volumes', {})
        # Check that the final volumes match the expected ones (workspace + non-conflicting custom)
        assert call_kwargs.get('volumes') == expected_volumes
        assert 'volumes' not in call_kwargs


def test_volume_merging_duplicate_custom(
    test_config, mock_event_stream, mock_docker_dependencies
):
    """Test duplicate host path definition in custom volumes. Should log warning."""
    # Disable workspace mount for simplicity
    test_config.workspace_mount_path = None
    test_config.workspace_mount_path_in_sandbox = None

    custom_volumes = {
        '/host/data': {'bind': '/data1', 'mode': 'ro'},
        '/host/data': {
            'bind': '/data2',
            'mode': 'rw',
        },  # Duplicate host path # noqa: F601
    }
    test_config.sandbox.docker_runtime_kwargs = json.dumps({'volumes': custom_volumes})

    with patch.object(logger, 'warning') as mock_log_warning:
        runtime = DockerRuntime(test_config, mock_event_stream)
        runtime._init_container()

        # Check if the warning about duplicate definition was logged
        # Note: The exact message might depend on dict iteration order, but check for the path.
        mock_log_warning.assert_any_call(
            'Duplicate volume definition for host path /host/data. Overwriting with custom volume definition.'
        )

        mock_docker_dependencies.containers.run.assert_called_once()
        call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
        # The behavior depends on dict iteration order during pop/merge.
        # The current implementation overwrites based on iteration order.
        # Let's assert the final state reflects one of the definitions.
        final_volumes = call_kwargs.get('volumes')
        assert '/host/data' in final_volumes
        # It should be the *last* definition encountered during iteration, but exact order isn't guaranteed.
        # We check if it's one of the valid definitions.
        assert final_volumes['/host/data'] == {
            'bind': '/data2',
            'mode': 'rw',
        } or final_volumes['/host/data'] == {'bind': '/data1', 'mode': 'ro'}
        # More robustly, we could check the warning was logged.
        assert 'volumes' not in call_kwargs


def test_volume_merging_invalid_format(
    test_config, mock_event_stream, mock_docker_dependencies
):
    """Test skipping invalid volume formats in custom volumes."""
    # Disable workspace mount for simplicity
    test_config.workspace_mount_path = None
    test_config.workspace_mount_path_in_sandbox = None

    custom_volumes = {
        '/host/valid': {'bind': '/valid_data', 'mode': 'rw'},
        '/host/invalid_string': '/invalid_data',  # Invalid format (string instead of dict)
        '/host/invalid_dict': {
            'source': '/invalid_data',
            'mode': 'rw',
        },  # Invalid format (missing 'bind')
    }
    test_config.sandbox.docker_runtime_kwargs = json.dumps({'volumes': custom_volumes})

    with patch.object(logger, 'warning') as mock_log_warning:
        runtime = DockerRuntime(test_config, mock_event_stream)
        runtime._init_container()

        # Check that warnings were logged for invalid formats
        mock_log_warning.assert_any_call(
            'Invalid volume format for host path /host/invalid_string in docker_runtime_kwargs. Skipping.'
        )
        mock_log_warning.assert_any_call(
            'Invalid volume format for host path /host/invalid_dict in docker_runtime_kwargs. Skipping.'
        )

        mock_docker_dependencies.containers.run.assert_called_once()
        call_args, call_kwargs = mock_docker_dependencies.containers.run.call_args
        # Only the valid volume should be present
        expected_volumes = {
            '/host/valid': {'bind': '/valid_data', 'mode': 'rw'},
        }
        assert call_kwargs.get('volumes') == expected_volumes
        assert 'volumes' not in call_kwargs
