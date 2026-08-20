"""
═════════════════════════════════════════════════════════════════════════════════
pitching_efficiency.py — Advanced Pitching Efficiency & Workload Weighting
═════════════════════════════════════════════════════════════════════════════════
Implements:
1. Recent FIP (Fielder Independent Pitching) weighting from last 3 starts
2. Bullpen Tax penalty for short-outing starters
3. Rolling workload analysis (IP per start)
4. Pitcher durability scoring
═══════════════════════════���═════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
FIP_CONSTANT = 3.20  # League average, can vary by season
BULLPEN_TAX_THRESHOLD = 5.67  # innings: if starter < 5.67 IP/start, apply penalty
BULLPEN_TAX_PENALTY_MIN = 0.06  # 6% penalty on win probability
BULLPEN_TAX_PENALTY_MAX = 0.08  # 8% penalty on win probability

# FIP formula: FIP = ((13 * HR) + (3 * BB) - (2 * K)) / IP + FIP_CONSTANT
# This requires granular pitcher data


class PitchingEfficiencyEngine:
    """
    Analyzes starting pitcher efficiency using:
    - Recent FIP over last 3 starts (weights recent performance)
    - Bullpen Tax: penalty when starter has history of short outings
    - Workload analysis and durability scoring
    """

    def __init__(self):
        self.fip_constant = FIP_CONSTANT
        self.bullpen_tax_threshold = BULLPEN_TAX_THRESHOLD
        self.bullpen_tax_min = BULLPEN_TAX_PENALTY_MIN
        self.bullpen_tax_max = BULLPEN_TAX_PENALTY_MAX
        logger.info("🎯 Pitching Efficiency Engine initialized")

    def compute_recent_fip(
        self,
        pitcher_data: Dict,
        last_n_starts: int = 3
    ) -> Tuple[float, Dict]:
        """
        Computes FIP (Fielder Independent Pitching) for the last N starts.
        
        Args:
            pitcher_data: Dictionary with pitcher's recent stats
                - 'last_starts': list of last game data dicts
                - Each entry needs: ip (innings pitched), hr, bb, k
            last_n_starts: how many recent starts to include
        
        Returns:
            (fip_score, metadata_dict)
        """
        metadata = {
            'starts_analyzed': 0,
            'total_ip': 0.0,
            'total_hr': 0,
            'total_bb': 0,
            'total_k': 0,
            'data_quality': 'unknown'
        }

        # Check if we have recent start data
        if not pitcher_data or 'last_starts' not in pitcher_data:
            logger.debug("⚠️ No recent start data for FIP calculation")
            metadata['data_quality'] = 'missing'
            # Return league average ERA-equivalent FIP
            return self.fip_constant, metadata

        last_starts = pitcher_data.get('last_starts', [])
        
        # Take only the last N starts
        recent_starts = last_starts[-last_n_starts:] if len(last_starts) > 0 else []
        
        if not recent_starts:
            metadata['data_quality'] = 'insufficient'
            return self.fip_constant, metadata

        # Aggregate stats
        total_ip = 0.0
        total_hr = 0
        total_bb = 0
        total_k = 0

        for start in recent_starts:
            if not isinstance(start, dict):
                continue
            
            try:
                ip = float(start.get('ip', 0))
                hr = int(start.get('hr', 0))
                bb = int(start.get('bb', 0))
                k = int(start.get('k', 0))
                
                total_ip += ip
                total_hr += hr
                total_bb += bb
                total_k += k
            except (ValueError, TypeError):
                continue

        metadata['starts_analyzed'] = len(recent_starts)
        metadata['total_ip'] = total_ip
        metadata['total_hr'] = total_hr
        metadata['total_bb'] = total_bb
        metadata['total_k'] = total_k

        if total_ip <= 0:
            metadata['data_quality'] = 'invalid'
            return self.fip_constant, metadata

        # FIP = ((13*HR + 3*BB - 2*K) / IP) + constant
        fip_numerator = (13 * total_hr) + (3 * total_bb) - (2 * total_k)
        fip = (fip_numerator / total_ip) + self.fip_constant

        metadata['data_quality'] = 'complete'
        metadata['fip'] = round(fip, 2)

        logger.debug(
            f"  FIP computed: {fip:.2f} | IP: {total_ip:.1f} | "
            f"HR: {total_hr} | BB: {total_bb} | K: {total_k}"
        )

        return fip, metadata

    def compute_workload_average(
        self,
        pitcher_data: Dict,
        last_n_starts: int = 5
    ) -> Tuple[float, Dict]:
        """
        Computes average innings pitched per start.
        
        Returns:
            (avg_ip_per_start, metadata)
        """
        metadata = {
            'starts_analyzed': 0,
            'avg_ip': 0.0,
            'min_ip': 0.0,
            'max_ip': 0.0,
            'data_quality': 'unknown'
        }

        if not pitcher_data or 'last_starts' not in pitcher_data:
            metadata['data_quality'] = 'missing'
            return 5.5, metadata

        last_starts = pitcher_data.get('last_starts', [])
        recent_starts = last_starts[-last_n_starts:] if len(last_starts) > 0 else []

        if not recent_starts:
            metadata['data_quality'] = 'insufficient'
            return 5.5, metadata

        ip_list = []
        for start in recent_starts:
            if isinstance(start, dict):
                try:
                    ip = float(start.get('ip', 0))
                    if ip > 0:
                        ip_list.append(ip)
                except (ValueError, TypeError):
                    continue

        if not ip_list:
            metadata['data_quality'] = 'invalid'
            return 5.5, metadata

        avg_ip = np.mean(ip_list)
        metadata['starts_analyzed'] = len(ip_list)
        metadata['avg_ip'] = round(avg_ip, 2)
        metadata['min_ip'] = round(np.min(ip_list), 1)
        metadata['max_ip'] = round(np.max(ip_list), 1)
        metadata['data_quality'] = 'complete'

        return avg_ip, metadata

    def calculate_bullpen_tax(
        self,
        pitcher_data: Dict,
        home_team: str,
        away_team: str
    ) -> Tuple[float, Dict]:
        """
        Applies Bullpen Tax penalty if starting pitcher averages < 5.67 IP/start.
        
        The penalty reduces the team's win probability by 6-8%, shifting value to opponent.
        
        Returns:
            (penalty_pct, metadata)
        """
        metadata = {
            'avg_ip': 0.0,
            'penalty_applied': False,
            'penalty_pct': 0.0,
            'threshold': self.bullpen_tax_threshold,
            'reasoning': ''
        }

        avg_ip, workload_meta = self.compute_workload_average(pitcher_data, last_n_starts=5)
        metadata['avg_ip'] = avg_ip

        if workload_meta['data_quality'] != 'complete':
            metadata['reasoning'] = f"Workload data quality: {workload_meta['data_quality']}"
            return 0.0, metadata

        # If average IP < threshold, apply penalty
        if avg_ip < self.bullpen_tax_threshold:
            # Scale penalty from 6% to 8% based on how far below threshold
            # Closer to 0 = higher penalty
            deficit = self.bullpen_tax_threshold - avg_ip
            max_deficit = self.bullpen_tax_threshold - 3.0  # Assume 3 IP is minimum

            if deficit <= 0:
                penalty_pct = self.bullpen_tax_min
            elif deficit >= max_deficit:
                penalty_pct = self.bullpen_tax_max
            else:
                # Linear interpolation
                penalty_pct = self.bullpen_tax_min + (
                    (deficit / max_deficit) * (self.bullpen_tax_max - self.bullpen_tax_min)
                )

            metadata['penalty_applied'] = True
            metadata['penalty_pct'] = round(penalty_pct, 4)
            metadata['reasoning'] = (
                f"Avg IP ({avg_ip:.2f}) < threshold ({self.bullpen_tax_threshold}). "
                f"Applying {penalty_pct*100:.1f}% win prob reduction to {home_team if away_team else away_team}."
            )

            logger.info(f"  ⚠️ Bullpen Tax: {metadata['reasoning']}")
        else:
            metadata['reasoning'] = (
                f"Starter durable: avg IP ({avg_ip:.2f}) >= threshold. No penalty."
            )

        return metadata['penalty_pct'], metadata

    def compute_durability_score(
        self,
        pitcher_data: Dict
    ) -> Tuple[float, Dict]:
        """
        Durability score (0-1) based on:
        - Consistency of IP across starts
        - Ability to go deep into games
        - Recent health trends
        
        Returns:
            (durability_score, metadata)
        """
        metadata = {
            'workload_consistency': 0.0,
            'deep_game_rate': 0.0,
            'durability_score': 0.0,
            'data_quality': 'unknown'
        }

        if not pitcher_data or 'last_starts' not in pitcher_data:
            metadata['data_quality'] = 'missing'
            return 0.5, metadata  # Neutral score

        last_starts = pitcher_data.get('last_starts', [])
        recent_starts = last_starts[-10:] if len(last_starts) >= 10 else last_starts

        if not recent_starts:
            metadata['data_quality'] = 'insufficient'
            return 0.5, metadata

        # Extract IP from each start
        ip_list = []
        deep_games = 0  # Starts with >= 6 IP

        for start in recent_starts:
            if isinstance(start, dict):
                try:
                    ip = float(start.get('ip', 0))
                    if ip > 0:
                        ip_list.append(ip)
                        if ip >= 6.0:
                            deep_games += 1
                except (ValueError, TypeError):
                    continue

        if not ip_list:
            metadata['data_quality'] = 'invalid'
            return 0.5, metadata

        # Workload consistency: std dev of IP (lower = more consistent)
        ip_std = np.std(ip_list)
        ip_mean = np.mean(ip_list)
        
        # Normalize std dev: assume typical std is 1.5
        consistency = max(0, 1 - (ip_std / 2.0))  # 0-1 scale

        # Deep game rate
        deep_game_rate = deep_games / len(ip_list)

        # Combined durability: weighted average
        durability_score = (consistency * 0.4) + (deep_game_rate * 0.6)
        durability_score = np.clip(durability_score, 0.0, 1.0)

        metadata['workload_consistency'] = round(consistency, 3)
        metadata['deep_game_rate'] = round(deep_game_rate, 3)
        metadata['durability_score'] = round(durability_score, 3)
        metadata['data_quality'] = 'complete'
        metadata['ip_mean'] = round(ip_mean, 2)
        metadata['ip_std'] = round(ip_std, 2)

        logger.debug(
            f"  Durability: {durability_score:.3f} | "
            f"Consistency: {consistency:.3f} | Deep Game %: {deep_game_rate*100:.1f}%"
        )

        return durability_score, metadata

    def apply_fip_adjustment_to_frame(
        self,
        df: pd.DataFrame,
        home_pitcher_col: str = 'home_starter_fip',
        away_pitcher_col: str = 'away_starter_fip',
        pitcher_data_source: Dict = None
    ) -> pd.DataFrame:
        """
        Applies FIP-based adjustments to a DataFrame of matchups.
        
        Assumes DataFrame has columns like:
        - home_team, away_team
        - home_starter_era, away_starter_era (existing)
        
        Adds columns:
        - home_recent_fip, away_recent_fip
        - fip_advantage
        - bullpen_tax_home, bullpen_tax_away
        """
        df = df.copy()

        # Placeholder columns
        df['home_recent_fip'] = self.fip_constant
        df['away_recent_fip'] = self.fip_constant
        df['fip_advantage'] = 0.0
        df['bullpen_tax_penalty'] = 0.0

        logger.info("✅ FIP adjustment columns added (awaiting live pitcher data)")

        return df

    def filter_confidence_by_fip_gap(
        self,
        df: pd.DataFrame,
        min_fip_gap: float = 0.75
    ) -> pd.DataFrame:
        """
        Variance reduction filter: only flag high-confidence picks if FIP gap is significant.
        
        Prevents overconfident picks when pitching is evenly matched.
        """
        df = df.copy()

        if 'fip_advantage' not in df.columns:
            logger.warning("⚠️ fip_advantage column not found")
            return df

        # Reduce confidence if FIP gap is small
        df['fip_confidence_gate'] = df['fip_advantage'].abs() >= min_fip_gap

        initial_confident = df['confident_pick'].sum()
        df.loc[~df['fip_confidence_gate'], 'confident_pick'] = False
        final_confident = df['confident_pick'].sum()

        reduction = initial_confident - final_confident
        logger.info(
            f"  FIP Confidence Gate: {initial_confident} → {final_confident} picks "
            f"(-{reduction} reduced due to small FIP gap)"
        )

        return df


# ─── Public API ──────────────────────────────────────────────────────────────

def create_pitching_efficiency_engine() -> PitchingEfficiencyEngine:
    """Factory function to create the pitching efficiency engine."""
    return PitchingEfficiencyEngine()


def validate_pitching_data(pitcher_data: Dict) -> bool:
    """Validates that pitcher data has minimum required fields."""
    if not pitcher_data or not isinstance(pitcher_data, dict):
        return False
    
    required = ['last_starts']
    return all(key in pitcher_data for key in required)
