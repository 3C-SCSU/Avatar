import pytest
from mock import MagicMock, patch


@pytest.fixture
def mock_session_services():
    with patch("robot_controller.qi.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_tts = MagicMock()
        mock_motion = MagicMock()
        mock_audio_player = MagicMock()

        mock_session.service.side_effect = lambda service_name: {
            "ALTextToSpeech": mock_tts,
            "ALMotion": mock_motion,
            "ALAudioPlayer": mock_audio_player,
        }[service_name]

        mock_session_cls.return_value = mock_session

        yield {
            "session": mock_session,
            "tts": mock_tts,
            "motion": mock_motion,
            "audio_player": mock_audio_player
        }
