"""Regression checks without loading model checkpoints or running a power flow."""
import ast
from pathlib import Path
import sys
import unittest

CODE = Path(__file__).resolve().parents[3] / 'paper_results/code'
sys.path.insert(0, str(CODE))
from smallcase_evaluation_contract import accepted, constraint_flags, primary_reason, verify_state


class ContractTest(unittest.TestCase):
    def test_ac_convergence_is_not_overall_acceptance(self):
        info = dict(converged=False, ac_converged=True, v_violations=2,
                    overloaded_lines=0, n1_violations=0, v_violation_max_pu=.01)
        self.assertFalse(accepted(info))
        self.assertEqual(primary_reason(info), 'voltage_violation')
        source = ast.parse((CODE / 'eval_case39_action_diagnosis.py').read_text(encoding='utf-8'))
        selected = [node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('_reason', '_is_pass')]
        context = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<classification>', 'exec'), context)
        self.assertTrue(context['_reason'](info).startswith('voltage_violation'))

    def test_multilabel_and_unavailable_metrics(self):
        info = dict(ac_converged=True, v_violations=2, overloaded_lines=1, n1_violations=3)
        flags = constraint_flags(info)
        self.assertTrue(flags['voltage_violation'] and flags['thermal_overload'] and flags['n1_violation'])
        info['ac_converged'] = False
        flags = constraint_flags(info)
        self.assertIsNone(flags['voltage_violation'])
        self.assertTrue(flags['n1_violation'])
        self.assertEqual(primary_reason(info), 'acpf_not_converged')

    def test_success_and_missing_checks(self):
        info = dict(ac_converged=True, v_violations=0, overloaded_lines=0, n1_violations=0)
        self.assertTrue(accepted(info))
        info['n1_checked'] = False
        self.assertFalse(accepted(info))
        self.assertEqual(primary_reason(info), 'incomplete_verification')

    def test_verification_does_not_execute_an_action(self):
        class Env:
            current_power = [1., 2.]
            dcnet = object()
            def step(self, action):
                raise AssertionError('Verification must not call step')
            def _run_ac_pf(self):
                return dict(converged=True, v_violations=0, overloaded_lines=0)
            def _check_n1_security(self, power, network):
                return 0, None
        env = Env()
        self.assertTrue(verify_state(env, None, 118)['passed'])
        self.assertEqual(env.current_power, [1., 2.])


if __name__ == '__main__':
    unittest.main()
