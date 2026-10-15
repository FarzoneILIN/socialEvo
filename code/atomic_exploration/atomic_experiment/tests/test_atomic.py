import copy
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from utils.client import ApiError, ChatClient, MockClient, Reply
from atomic_experiment.config import Config
from atomic_experiment.environment import Environment, make_task, parse_action
from atomic_experiment.recording import Recorder
from atomic_experiment.runner import episode, run


class EnvironmentTests(unittest.TestCase):
    def test_unique_intersection(self):
        for seed in range(100):
            task = make_task(seed)
            self.assertEqual(set(task.clues[0]) & set(task.clues[1]), {task.answer})
            self.assertEqual(len(set(task.candidates)), 16)
            self.assertEqual([len(c) for c in task.clues], [4, 4])
            self.assertEqual(task, make_task(seed))

    def test_private_observation_and_copy(self):
        env = Environment(make_task(42), True)
        obs = env.observe(0, 1, 4)
        self.assertNotIn('answer', obs)
        self.assertNotIn('clues', obs)
        self.assertEqual(obs['private_possible_answers'], list(env.task.clues[0]))
        obs['inbox'].append('corruption')
        self.assertEqual(env.inboxes[0], [])

    def test_disabled_channel_and_invalid_actions(self):
        env = Environment(make_task(1), False)
        for action in ({'type': 'send', 'content': 'hello'}, {'type': 'unknown'},
                       {'type': 'wait', 'extra': 1}, {'type': 'submit', 'answer': []}):
            self.assertFalse(env.apply(0, action)['accepted'])
        self.assertEqual(env.inboxes, [[], []])

    def test_submission_has_no_correctness_feedback(self):
        env = Environment(make_task(1), True)
        self.assertEqual(env.apply(0, {'type': 'submit', 'answer': env.task.answer}),
                         {'accepted': True, 'reason': 'accepted'})
        self.assertFalse(env.apply(0, {'type': 'wait'})['accepted'])
        self.assertNotIn('correct', env.observe(1, 2, 4))
        self.assertFalse(env.score()['team_success'])

    def test_json_validation(self):
        for raw in ('[]', 'not json', '{"type":"wait","type":"send"}'):
            with self.assertRaises(ValueError):
                parse_action(raw)


class RunnerTests(unittest.TestCase):
    def execute(self, config, client):
        with tempfile.TemporaryDirectory() as root:
            recorder = Recorder(Path(root))
            result = episode(config, client, make_task(42), True, 'test', recorder)
            events = [json.loads(x) for x in (Path(root) / 'events.jsonl').read_text().splitlines()]
        return result, events

    def test_synchronous_rounds_and_mock_success(self):
        result, events = self.execute(Config(), MockClient())
        self.assertTrue(result['team_success'])
        self.assertEqual(result['calls'], 4)
        first = [e for e in events if e['kind'] == 'request' and e['round'] == 1]
        self.assertEqual(len(first), 2)
        self.assertTrue(all(json.loads(e['messages'][-1]['content'])['inbox'] == [] for e in first))
        self.assertIsNone(result['reported_tokens'])

    def test_call_budget(self):
        result, _ = self.execute(Config(max_calls=1), MockClient())
        self.assertEqual(result['calls'], 1)
        self.assertEqual(result['status'], 'budget_limit')

    def test_invalid_actions_consume_budget(self):
        class BadClient:
            def complete(self, messages):
                return Reply('bad json', None, 'test')
        result, _ = self.execute(Config(rounds=2), BadClient())
        self.assertEqual(result['invalid_actions'], 4)
        self.assertEqual(result['status'], 'round_limit')

    def test_missing_usage_fails_closed(self):
        result, _ = self.execute(Config(backend='api'), MockClient())
        self.assertEqual(result['calls'], 1)
        self.assertEqual(result['status'], 'usage_unknown')

    def test_reported_token_stop(self):
        class MeteredClient(MockClient):
            def complete(self, messages):
                reply = super().complete(messages)
                reply.usage = {'total_tokens': 20}
                return reply
        result, _ = self.execute(Config(backend='api', reported_token_limit=10), MeteredClient())
        self.assertEqual(result['calls'], 1)
        self.assertEqual(result['reported_tokens'], 20)
        self.assertEqual(result['status'], 'budget_limit')

    def test_batch_pairing(self):
        with tempfile.TemporaryDirectory() as root:
            directory, aborted = run(Config(), MockClient(), root)
            summary = json.loads((directory / 'summary.json').read_text())
            self.assertFalse(aborted)
            self.assertEqual(len(summary['paired_differences']), 4)
            self.assertEqual(summary['groups']['comm']['team_success_rate'], 1)
            self.assertIsNone(summary['groups']['comm']['total_reported_tokens'])

    def test_batch_aborts_on_transport_failure(self):
        class BrokenClient(MockClient):
            def complete(self, messages):
                raise ApiError('transport_error_billing_unknown')
        with tempfile.TemporaryDirectory() as root:
            directory, aborted = run(Config(backend='api'), BrokenClient(), root)
            summary = json.loads((directory / 'summary.json').read_text())
            self.assertTrue(aborted)
            self.assertEqual(summary['groups']['comm']['evaluable'], 0)
            self.assertEqual(summary['groups']['no_comm']['attempted'], 0)

    def test_secret_redaction(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = Recorder(Path(root), 'secret-test-token')
            recorder.event('e', 'response', content='echo secret-test-token')
            self.assertNotIn('secret-test-token', (Path(root) / 'events.jsonl').read_text())


class AdapterTests(unittest.TestCase):
    def test_local_http_adapter(self):
        captured = {}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                captured['path'] = self.path
                captured['auth'] = self.headers.get('Authorization')
                captured['body'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                body = json.dumps({'choices': [{'message': {'content': '{"type":"wait"}'},
                                               'finish_reason': 'stop'}],
                                   'usage': {'total_tokens': 12}, 'model': 'local-stub'}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {'SOCIALEVO_BASE_URL': f'http://127.0.0.1:{server.server_port}/v1',
                                         'SOCIALEVO_API_KEY': 'test-only', 'SOCIALEVO_MODEL': 'stub'}):
                client = ChatClient(Config(backend='api'))
                reply = client.complete([{'role': 'user', 'content': 'test'}])
            self.assertEqual(reply.usage['total_tokens'], 12)
            self.assertEqual(captured['path'], '/v1/chat/completions')
            self.assertEqual(captured['auth'], 'Bearer test-only')
            self.assertEqual(captured['body']['max_tokens'], 256)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                ChatClient(Config(backend='api'))


if __name__ == '__main__':
    unittest.main()
