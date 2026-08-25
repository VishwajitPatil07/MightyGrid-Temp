import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.brain.planner import _heuristic_intent, plan_steps
from src.brain.vllm_client import parse_ui_tars_action
from src.eyes.screen import clickable_still_visible, deterministic_button, parse_elements, point_hits_element
from src.hands import adb_client
from src.hands.maatouch import MaaTouch
from src.hands.personas import generate_persona
from scripts.run_poc import is_verified_completion, perceive, run_device_safely


class PlannerTests(unittest.TestCase):
    def test_heuristic_intent_extracts_app_and_credentials(self):
        intent = _heuristic_intent(
            "Open the Agoda app, enter email test@example.com and password: Secret123!"
        )

        self.assertEqual(intent["app"], "Agoda")
        self.assertEqual(intent["values"], ["test@example.com", "Secret123!"])

    def test_heuristic_intent_extracts_flight_route_and_date(self):
        intent = _heuristic_intent(
            "Open MakeMyTrip and search for a flight from Delhi to Dubai on 21 Dec 2026."
        )

        self.assertEqual(intent["app"], "MakeMyTrip")
        self.assertEqual(intent["values"], ["Delhi", "Dubai"])

    def test_plan_steps_preserves_order_without_splitting_credential_pairs(self):
        steps = plan_steps(
            "Open the Agoda app and enter email test@example.com and password: Secret123! "
            "then search for hotels in Tokyo"
        )

        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["values"], ["test@example.com", "Secret123!"])
        self.assertEqual(steps[1]["values"], ["hotels in Tokyo"])


class ActionParsingTests(unittest.TestCase):
    def test_parses_supported_actions(self):
        self.assertEqual(
            parse_ui_tars_action("Action: click(start_box='(123, 456)')"),
            {"action": "click", "target": [123, 456]},
        )
        self.assertEqual(
            parse_ui_tars_action("Action: scroll(direction='up')"),
            {"action": "scroll", "direction": "up"},
        )
        self.assertEqual(
            parse_ui_tars_action("Action: finished()"),
            {"action": "finished"},
        )

    def test_unrecognized_action_is_safe_noop(self):
        self.assertEqual(
            parse_ui_tars_action("Action: make_purchase()"),
            {"action": "none", "target": "error"},
        )


class WorkflowOutcomeTests(unittest.TestCase):
    def test_only_evidence_backed_completion_is_successful(self):
        self.assertTrue(is_verified_completion("completed"))
        self.assertFalse(is_verified_completion("completed_unverified"))
        self.assertFalse(is_verified_completion("abandoned"))

    def test_device_worker_reports_normal_completion(self):
        with patch("scripts.run_poc.run_workflow") as run_workflow:
            self.assertEqual(run_device_safely("device-1", "open an app", 42), "finished")

        run_workflow.assert_called_once_with("device-1", "open an app", 42)

    def test_device_worker_contains_unexpected_exception(self):
        with patch("scripts.run_poc.run_workflow", side_effect=RuntimeError("connection lost")):
            self.assertEqual(run_device_safely("device-1", "open an app"), "error")


class ScreenHelperTests(unittest.TestCase):
    def test_parse_elements_reads_clickable_bounds_and_label(self):
        xml = (
            '<hierarchy><node text="Continue" content-desc="" '
            'class="android.widget.Button" clickable="true" password="false" '
            'bounds="[10,20][110,70]" /></hierarchy>'
        )

        self.assertEqual(
            parse_elements(xml),
            [{
                "label": "Continue",
                "text": "Continue",
                "desc": "",
                "class": "android.widget.Button",
                "rid": "",
                "focused": False,
                "password": False,
                "x1": 10,
                "y1": 20,
                "x2": 110,
                "y2": 70,
                "cx": 60,
                "cy": 45,
                "w": 100,
                "h": 50,
                "clickable": True,
            }],
        )

    def test_deterministic_button_waits_until_values_are_typed(self):
        elements = [{"clickable": True, "label": "Continue"}]

        self.assertIsNone(deterministic_button("continue", elements, ["value"], 0))
        self.assertIs(deterministic_button("continue", elements, ["value"], 1), elements[0])

    def test_deterministic_button_does_not_submit_a_dated_flight_route(self):
        elements = [{"clickable": True, "label": "SEARCH FLIGHTS"}]

        self.assertIsNone(
            deterministic_button(
                "search for a flight from Delhi to Dubai on 21 Dec 2026",
                elements,
                ["Delhi", "Dubai"],
                2,
            )
        )

    def test_clickable_still_visible_detects_an_unchanged_submit_button(self):
        before = parse_elements(
            '<hierarchy><node text="Log in" content-desc="" class="android.widget.Button" '
            'clickable="true" password="false" bounds="[10,120][210,180]" /></hierarchy>'
        )[0]
        after = parse_elements(
            '<hierarchy><node text="Log in" content-desc="" class="android.widget.Button" '
            'clickable="true" password="false" bounds="[12,121][212,181]" /></hierarchy>'
        )

        self.assertTrue(clickable_still_visible(before, after))
        self.assertTrue(point_hits_element(before, 205, 175))
        self.assertFalse(point_hits_element(before, 300, 300))


class MaaTouchTests(unittest.TestCase):
    def test_maatouch_is_default_and_u2_can_be_forced(self):
        with patch.object(adb_client, "_maatouch", object()):
            with patch.dict(adb_client.os.environ, {}, clear=True):
                self.assertTrue(adb_client._use_maatouch())
            with patch.dict(adb_client.os.environ, {"MG_TOUCH": "u2"}, clear=True):
                self.assertFalse(adb_client._use_maatouch())

    def test_maatouch_tap_error_falls_back_cleanly(self):
        persona = SimpleNamespace()
        with patch.object(adb_client, "_maatouch") as maatouch, \
             patch.dict(adb_client.os.environ, {}, clear=True), \
             patch.object(adb_client, "_mt_pressure", return_value=64):
            maatouch.get.side_effect = OSError("daemon blip")
            self.assertFalse(adb_client._mt_tap("device-1", 10, 20, 0.05, persona))

    def test_long_press_sends_drift_and_remaining_hold_on_device(self):
        touch = MaaTouch.__new__(MaaTouch)
        touch.max_p = 255
        with patch.object(touch, "alive", return_value=True), patch.object(touch, "_write") as write:
            self.assertTrue(touch.long_press(10, 20, 700, [(11, 21, 250)], pressure=64))

        write.assert_called_once_with([
            "d 0 10 20 64", "w 250", "m 0 11 21 64", "w 450", "u 0", "c",
        ])

    def test_adb_long_press_routes_human_drift_to_maatouch(self):
        touch = MagicMock()
        touch.long_press.return_value = True
        persona = SimpleNamespace(tremor_scale=1.0)

        with patch.object(adb_client, "_maatouch") as maatouch, \
             patch.dict(adb_client.os.environ, {}, clear=True), \
             patch.object(adb_client, "get_device_size", return_value=(1080, 2400)), \
             patch.object(adb_client, "_mt_pressure", return_value=64), \
             patch.object(adb_client.random, "gauss", return_value=0.0):
            maatouch.get.return_value = touch
            self.assertTrue(adb_client._mt_long_press("device-1", 10, 20, 0.48, persona))

        touch.long_press.assert_called_once_with(
            10, 20, 480, [(10, 20, 120)] * 4, 64,
        )


class PerceptionTests(unittest.TestCase):
    def test_perceive_retries_a_transient_screenshot_failure(self):
        device = MagicMock()
        device.dump_hierarchy.return_value = "<hierarchy />"
        device.screenshot.side_effect = [OSError("capture blip"), None]

        with patch("scripts.run_poc.get_device_for_serial", return_value=device), \
             patch("scripts.run_poc.time.sleep"):
            xml, path, elements = perceive("device-1", attempts=2, retry_delay=0)

        self.assertEqual(xml, "<hierarchy />")
        self.assertEqual(path, "screen_device-1.png")
        self.assertEqual(elements, [])
        self.assertEqual(device.screenshot.call_count, 2)
        self.assertEqual(device.dump_hierarchy.call_count, 2)


class PersonaTests(unittest.TestCase):
    def test_seeded_persona_is_reproducible(self):
        first = generate_persona(20260821)
        second = generate_persona(20260821)

        self.assertEqual(first, second)
        self.assertEqual(first.archetype, "careful")

    def test_unknown_archetype_uses_average(self):
        self.assertEqual(generate_persona(7, archetype="unknown").archetype, "average")


if __name__ == "__main__":
    unittest.main()