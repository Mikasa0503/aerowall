import unittest
from aerowall.rally_events import ContactLedger, Impact, Kind, RallyBatch, RallyState

T=(0.,0.,2.)
def cap():return Impact(Kind.CAP,('ball','bat'),1.,(0.,0.,1.))
def wall(point=T,eligible=True):return Impact(Kind.WALL,('ball','wall'),1.,point,eligible)

class RallyContract(unittest.TestCase):
    def test_reset_epoch_persist_is_uncredited_until_positive_impulse(self):
        from aerowall.rally_events import ContactLedger, Kind
        ledger=ContactLedger();pair=('ball','bat')
        self.assertIsNone(ledger.observe(pair,'persist',Kind.CAP,0.,episode_start=True))
        self.assertIsNotNone(ledger.observe(pair,'persist',Kind.CAP,.1))
        self.assertIsNone(ledger.observe(pair,'persist',Kind.CAP,.1))
        ledger.observe(pair,'lost',Kind.CAP)
        with self.assertRaises(RuntimeError):
            ledger.observe(pair,'persist',Kind.CAP,.1)
        ledger.reset()
        self.assertIsNotNone(ledger.observe(pair,'persist',Kind.BALL_BODY,.1,episode_start=True))

    def test_chained_rallies_and_target_publication(self):
        s=RallyState()
        self.assertFalse(s.advance([wall()],T)['rally_completed'])
        s.advance([cap()],T)
        out=s.advance([wall()],T)
        self.assertEqual(out['publish_next_target'],1)
        self.assertEqual(s.phase,'to_bat')
        out=s.advance([cap()],(0.,4.,2.))
        self.assertTrue(out['joint_completed']) # Uses old target at wall contact.
        self.assertEqual(out['completed_target_error'],0.)
        s.advance([wall()],(0.,4.,2.)) # physical rally, target miss
        out=s.advance([cap()],T)
        self.assertTrue(out['rally_completed'])
        self.assertFalse(out['joint_completed'])
        self.assertEqual((s.rallies,s.streak,s.max_streak,s.joint_rallies,s.joint_streak),(2,2,2,1,0))

    def test_double_wall_and_extra_bat_break_streak(self):
        for extra in [cap(),wall()]:
            s=RallyState();s.advance([cap()],T);s.advance([wall()],T);s.advance([cap()],T)
            if extra.kind==Kind.CAP:
                s.advance([extra],T)
            else:
                s.advance([wall()],T);s.advance([extra],T)
            self.assertEqual(s.streak,0)
            self.assertEqual(s.rallies,1)
            self.assertEqual(s.max_streak,1)
            self.assertFalse(s.terminated)

    def test_event_lifecycle_delayed_impulse_and_duplicates(self):
        l=ContactLedger();pair=('ball','bat')
        self.assertIsNone(l.observe(pair,'found',Kind.CAP,0.))
        self.assertIsNotNone(l.observe(pair[::-1],'persist',Kind.CAP,.01))
        for edge in ['persist','found','persist']:
            self.assertIsNone(l.observe(pair,edge,Kind.CAP,.1))
        l.observe(pair,'lost',Kind.CAP)
        self.assertIsNotNone(l.observe(pair,'found',Kind.CAP,.1))
        l.reset()
        with self.assertRaises(RuntimeError):l.observe(pair,'persist',Kind.CAP,.1)

    def test_manifold_sliding_off_cap_remains_illegal(self):
        l=ContactLedger();pair=('ball','bat')
        self.assertIsNotNone(l.observe(pair,'found',Kind.CAP,.1))
        bad=l.observe(pair,'persist',Kind.NON_CAP,.01)
        s=RallyState();s.advance([cap()],T);s.advance([wall()],T)
        out=s.advance([cap(),bad],T,time_limit=True)
        self.assertTrue(s.terminated);self.assertFalse(s.truncated)
        self.assertFalse(out['rally_completed']);self.assertEqual(s.rallies,0)
        self.assertTrue(s.advance([cap()],T)['ignored_after_done'])

    def test_ambiguous_order_is_not_success(self):
        for impacts in [[cap(),wall()],[wall(),cap()],[cap(),cap()],[wall(),wall()]]:
            s=RallyState();s.advance([cap()],T);s.advance([wall()],T)
            self.assertTrue(s.advance(impacts,T)['ambiguous_order'])
            self.assertEqual((s.rallies,s.phase),(0,'wait_bat'))

    def test_selective_reset_clears_contact_and_counters(self):
        b=RallyBatch(2)
        for s,l in zip(b.states,b.ledgers):
            s.advance([cap()],T);s.advance([wall()],T)
            l.observe(('ball','bat'),'found',Kind.CAP,.1)
        b.reset([0])
        self.assertEqual((b.states[0].phase,b.states[0].target_index,b.states[0].steps),('wait_bat',0,0))
        self.assertFalse(b.ledgers[0].active)
        self.assertEqual(b.states[1].phase,'to_bat');self.assertTrue(b.ledgers[1].active)
        self.assertTrue(b.states[1].advance([cap()],T)['rally_completed'])
        with self.assertRaises(IndexError):b.reset([-1])

    def test_timeout_and_failure_are_distinct(self):
        s=RallyState();s.advance([cap()],T);s.advance([wall()],T)
        self.assertTrue(s.advance([cap()],T,time_limit=True)['rally_completed'])
        self.assertTrue(s.truncated);self.assertFalse(s.terminated)
        for cause in ['numerical_failure','out_of_bounds']:
            s=RallyState();s.advance([],T,failure=cause,time_limit=True)
            self.assertTrue(s.terminated);self.assertFalse(s.truncated);self.assertEqual(s.reason,cause)

    def test_target_face_and_radius(self):
        for point,eligible,expected in [(T,False,False),((0.,.35,2.),True,True),((0.,.351,2.),True,False)]:
            s=RallyState();s.advance([cap()],T);s.advance([wall(point,eligible)],T)
            self.assertEqual(s.advance([cap()],T)['joint_completed'],expected)

    def test_invalid_sensing_data_is_rejected(self):
        l=ContactLedger()
        for value in [float('nan'),float('inf'),-.1]:
            with self.assertRaises(ValueError):l.observe(('ball','wall'),'found',Kind.WALL,value,T)
        with self.assertRaises(ValueError):RallyState().advance([wall((0.,float('nan'),2.))],T)

if __name__=='__main__':unittest.main()
