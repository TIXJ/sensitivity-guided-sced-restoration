import unittest
from analyze_deployment_accounting import analyze,route


class AccountingTests(unittest.TestCase):
    def payload(self):
        rows=[]
        for repeat in range(3):
            for sid in ('common','direct_only'):
                for arm in ('direct','selected'):
                    rows.append({'sample_id':sid,'repeat':repeat,'arm':arm,
                        'passed':sid=='common' or arm=='direct','preparation_failed':False,
                        'initial_passed':False,'policy_attempted':arm=='selected' and sid=='common',
                        'optimizer_called':not(arm=='selected' and sid=='common'),
                        'policy_steps':4 if arm=='selected' and sid=='common' else 0,
                        'policy_sec':.2,'runtime_sec':1.,'initial_features':{'demand_mw':16000.},
                        'scheduled_cost':10. if arm=='direct' else 12.,
                        'final_verification':{'reference_source_p_mw':2. if arm=='direct' else -1.}})
        return {'rows':rows}

    def test_pair_only_common_ids_and_signed_export(self):
        result=analyze(self.payload())
        for item in result['signed_exchange_per_repeat']:
            self.assertEqual(item['common_accepted_ids'],['common'])
            self.assertEqual(item['prices'][1]['direct_mean_total_cost'],60.)
            self.assertEqual(item['prices'][1]['selected_mean_total_cost'],-13.)
            self.assertEqual(item['prices'][1]['paired_selected_minus_direct'],-73.)

    def test_empty_failed_attempt_group_is_not_zero_time_estimate(self):
        result=analyze(self.payload())
        group=result['groups']['policy_fallback']
        self.assertEqual(group['count_per_repeat'],[0,0,0])
        self.assertIsNone(group['mean_selected_sec'])

    def test_route_priority_and_fallback(self):
        row={'preparation_failed':True}
        self.assertEqual(route(row),'preparation_failed')
        row={'policy_attempted':True,'optimizer_called':True}
        self.assertEqual(route(row),'policy_fallback')


if __name__=='__main__':
    unittest.main()
