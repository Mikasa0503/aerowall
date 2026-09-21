"""Project-owned return-protocol correction for the pinned recurrent Actor.

Preserves module/parameter paths; does not change GRU mathematics or carry
hidden state through the collector. That integration needs separate validation.
"""


class RecurrentActorReturnMixin:
    def forward(self, *args, **kwargs):
        result=super().forward(*args, **kwargs)
        if self.rnn is None:
            raise ValueError('Recurrent protocol correction requires an RNN actor')
        # Actor.forward's sixth positional argument is eval_action.
        eval_action=kwargs.get('eval_action', args[5] if len(args)>5 else False)
        if eval_action:
            if len(result)!=6 or result[3] is not None:
                raise RuntimeError('Pinned recurrent evaluation return protocol changed')
            corrected=result
        else:
            if len(result)!=7 or result[3] is not None:
                raise RuntimeError('Pinned recurrent rollout return protocol changed')
            corrected=result[:3]+result[4:]
        return corrected if self.output_dist_params else corrected[:4]


_adapted_classes={}


def adapt_recurrent_actor(actor):
    if actor.rnn is None:
        raise ValueError('Expected recurrent Actor')
    if isinstance(actor,RecurrentActorReturnMixin):
        return actor
    original=type(actor)
    if original not in _adapted_classes:
        _adapted_classes[original]=type('Corrected'+original.__name__,(RecurrentActorReturnMixin,original),{})
    actor.__class__=_adapted_classes[original]
    return actor
