from __future__ import annotations

import pytest

from marketingos.agents.business_analysis import BusinessAnalysisAgentConfig
from marketingos.agents.strategist import StrategistAgentConfig
from marketingos.prompts.registry import PromptRegistry

# Stable substrings, deliberately not full snapshots: these assert the *meaning*
# that must hold, so wording tweaks stay cheap while a real regression (a variant
# that stops composing the base, or a template wired to the wrong policy) fails.

#: Core of the shared guardrail — facts and assumptions are never blended.
BASE_CORE_SENTENCE = "must never be blended"

#: The strict, fact-only grounding rule used by business_analysis.
STRICT_CITATION = "at least one cited fact identifier"

#: The flexible rule permitting fact or assumption citations, used by strategist.
FLEXIBLE_CITATION = "fact (F*) or assumption (A*) identifier"

#: Citation notation for a labelled assumption id.
ASSUMPTION_MARKER = "A*"

#: Citation notation for an observed fact id.
FACT_MARKER = "F*"


@pytest.fixture(scope="module")
def registry() -> PromptRegistry:
    """A registry rooted at the real prompt library shipped with the package."""
    return PromptRegistry()


# ---------------------------------------------------------------------------
# 1. Strict variant: fact-only citation
# ---------------------------------------------------------------------------


def test_strict_policy_requires_a_fact_citation(registry: PromptRegistry):
    """business_analysis's variant demands grounding in an observed fact."""
    strict = registry.get_policy("facts_vs_assumptions")
    assert STRICT_CITATION in strict


def test_strict_policy_does_not_permit_assumption_citations(
    registry: PromptRegistry,
):
    """The strict variant must not open the door to A* grounding."""
    strict = registry.get_policy("facts_vs_assumptions")
    assert ASSUMPTION_MARKER not in strict
    assert FLEXIBLE_CITATION not in strict


# ---------------------------------------------------------------------------
# 2. Flexible variant: fact or assumption citation
# ---------------------------------------------------------------------------


def test_flexible_policy_permits_fact_and_assumption_citations(
    registry: PromptRegistry,
):
    """strategist's variant accepts either kind of identifier."""
    flexible = registry.get_policy("grounding_flexible")
    assert FLEXIBLE_CITATION in flexible
    assert FACT_MARKER in flexible
    assert ASSUMPTION_MARKER in flexible


def test_flexible_policy_omits_the_fact_only_constraint(registry: PromptRegistry):
    """The contradiction this split exists to remove must stay absent."""
    flexible = registry.get_policy("grounding_flexible")
    assert STRICT_CITATION not in flexible


# ---------------------------------------------------------------------------
# 3. Composition: both variants are built from the shared base
# ---------------------------------------------------------------------------


def test_both_variants_contain_the_shared_base_core(registry: PromptRegistry):
    """Neither variant may drop the never-blend guarantee."""
    for policy_name in ("facts_vs_assumptions", "grounding_flexible"):
        assert BASE_CORE_SENTENCE in registry.get_policy(policy_name)


def test_variants_compose_the_base_rather_than_duplicating_it(
    registry: PromptRegistry,
):
    """The whole rendered base must appear verbatim inside both variants.

    Compared against the base as loaded at runtime rather than a frozen
    literal, so rewording the base stays a one-file change — but two files
    drifting apart (a copy-paste instead of an ``{% include %}``) fails here.
    """
    base = registry.get_policy("grounding_base").strip()
    assert base, "grounding_base must not be empty"
    assert base in registry.get_policy("facts_vs_assumptions")
    assert base in registry.get_policy("grounding_flexible")


def test_shared_base_carries_no_citation_rule(registry: PromptRegistry):
    """The base is the non-contradictory core: it takes no side on citations."""
    base = registry.get_policy("grounding_base")
    assert BASE_CORE_SENTENCE in base
    assert STRICT_CITATION not in base
    assert FLEXIBLE_CITATION not in base


# ---------------------------------------------------------------------------
# 4. Templates render the correct policy (asserted through render(), so a
#    template rewired to the wrong policy fails here too)
# ---------------------------------------------------------------------------


def test_business_analysis_system_renders_the_strict_policy(
    registry: PromptRegistry,
):
    """The analysis system prompt carries the fact-only guardrail."""
    rendered = registry.render("business_analysis/v1/system")
    assert BASE_CORE_SENTENCE in rendered
    assert STRICT_CITATION in rendered
    assert FLEXIBLE_CITATION not in rendered


def test_strategist_first_week_system_renders_the_flexible_policy(
    registry: PromptRegistry,
):
    """The strategist system prompt carries the F*-or-A* guardrail."""
    rendered = registry.render("strategist/v1/first_week_system")
    assert BASE_CORE_SENTENCE in rendered
    assert FLEXIBLE_CITATION in rendered
    assert STRICT_CITATION not in rendered


# ---------------------------------------------------------------------------
# 5. Each agent's *configured* default resolves to the right guardrail.
#
#    The reference is read off the config object rather than repeated as a
#    literal, so repointing ``system_prompt_template`` anywhere else fails
#    here without the test needing to know where "anywhere else" is: a
#    nonexistent reference makes render() raise, and a real-but-wrong one
#    trips the policy assertions. These also exercise the agents' real
#    two-segment references, and so the registry's default-version
#    resolution, which the v1-pinned tests above deliberately bypass.
# ---------------------------------------------------------------------------


def test_business_analysis_configured_default_renders_strict_grounding(
    registry: PromptRegistry,
):
    """BusinessAnalysisAgent's configured prompt must ground in facts only.

    Asserted as "the strict rule is present and A* grounding is absent"
    rather than as the literal ``F*`` marker: the strict policy states the
    constraint in prose ("at least one cited fact identifier") and never uses
    the ``F*`` notation, so requiring that marker would fail on correct text.
    """
    reference = BusinessAnalysisAgentConfig().system_prompt_template
    rendered = registry.render(reference)
    assert BASE_CORE_SENTENCE in rendered
    assert STRICT_CITATION in rendered
    assert ASSUMPTION_MARKER not in rendered
    assert FLEXIBLE_CITATION not in rendered


def test_strategist_configured_default_renders_flexible_grounding(
    registry: PromptRegistry,
):
    """StrategistAgent's configured prompt must permit F* or A* grounding."""
    reference = StrategistAgentConfig().system_prompt_template
    rendered = registry.render(reference)
    assert BASE_CORE_SENTENCE in rendered
    assert FLEXIBLE_CITATION in rendered
    assert FACT_MARKER in rendered
    assert ASSUMPTION_MARKER in rendered
    assert STRICT_CITATION not in rendered
