import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional, Set
import logging
from collections import defaultdict, Counter
from itertools import combinations
import face_recognition
from .improved_face_detector import FaceInfo, FaceQuality

class AdaptiveThreshold:
    """
    Manages adaptive thresholds based on face quality and confidence.
    """
    
    def __init__(self):
        # Quality-based thresholds for face matching
        self.quality_thresholds = {
            FaceQuality.EXCELLENT: 0.35,  # Strictest for high-quality faces
            FaceQuality.GOOD: 0.45,
            FaceQuality.FAIR: 0.55,
            FaceQuality.POOR: 0.65        # Most lenient for poor-quality faces
        }
        
        # Confidence-based adjustments
        self.confidence_adjustments = {
            'high_confidence': -0.05,     # Stricter for high confidence
            'medium_confidence': 0.0,
            'low_confidence': 0.1         # More lenient for low confidence
        }

    def get_threshold(self, face1: FaceInfo, face2: FaceInfo) -> float:
        """
        Calculate adaptive threshold for comparing two faces.
        """
        # Use the higher quality level for threshold (stricter)
        quality_level = max(face1.quality_level, face2.quality_level, key=lambda x: x.value)
        base_threshold = self.quality_thresholds[quality_level]
        
        # Adjust based on confidence
        avg_confidence = (face1.confidence + face2.confidence) / 2
        if avg_confidence > 0.8:
            confidence_adj = self.confidence_adjustments['high_confidence']
        elif avg_confidence > 0.5:
            confidence_adj = self.confidence_adjustments['medium_confidence']
        else:
            confidence_adj = self.confidence_adjustments['low_confidence']
        
        return base_threshold + confidence_adj

class ImprovedFaceClusterer:
    """
    Advanced face clustering with hierarchical approach, adaptive thresholds, 
    and quality-based confidence scoring.
    """

    def __init__(self, min_cluster_size: int = 1, enable_hierarchical: bool = True):
        """
        Initialize the improved face clusterer.
        
        Args:
            min_cluster_size: Minimum number of faces to form a cluster
            enable_hierarchical: Whether to use hierarchical clustering approach
        """
        self.min_cluster_size = min_cluster_size
        self.enable_hierarchical = enable_hierarchical
        self.adaptive_threshold = AdaptiveThreshold()
        self.scaler = StandardScaler()

    def cluster_faces_advanced(self, face_infos: List[FaceInfo]) -> Tuple[np.ndarray, int, Dict]:
        """
        Advanced face clustering with multiple strategies and quality awareness.
        
        Returns:
            - cluster_labels: Array of cluster assignments
            - num_clusters: Number of clusters found
            - cluster_info: Dictionary with detailed cluster information
        """
        if not face_infos:
            return np.array([]), 0, {}
        
        # Phase 1: Quality-based pre-filtering and grouping
        quality_groups = self._group_by_quality(face_infos)
        
        # Phase 2: Multi-stage clustering
        if self.enable_hierarchical:
            cluster_labels, cluster_info = self._hierarchical_clustering(face_infos, quality_groups)
        else:
            cluster_labels, cluster_info = self._adaptive_dbscan_clustering(face_infos)
        
        # Phase 3: Post-processing and validation
        cluster_labels, cluster_info = self._post_process_clusters(face_infos, cluster_labels, cluster_info)
        
        num_clusters = len(set(cluster_labels)) if len(cluster_labels) > 0 else 0
        
        return cluster_labels, num_clusters, cluster_info

    def _group_by_quality(self, face_infos: List[FaceInfo]) -> Dict[FaceQuality, List[int]]:
        """
        Group faces by quality level for targeted processing.
        """
        quality_groups = defaultdict(list)
        for i, face_info in enumerate(face_infos):
            quality_groups[face_info.quality_level].append(i)
        return dict(quality_groups)

    def _hierarchical_clustering(self, face_infos: List[FaceInfo], quality_groups: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Hierarchical clustering approach: start with high-quality faces, then merge others.
        """
        n_faces = len(face_infos)
        cluster_labels = np.full(n_faces, -1)  # -1 means unassigned
        cluster_info = {}
        current_cluster_id = 0
        
        # Step 1: Process high-quality faces first (most reliable)
        high_quality_faces = []
        for quality in [FaceQuality.EXCELLENT, FaceQuality.GOOD]:
            if quality in quality_groups:
                high_quality_faces.extend(quality_groups[quality])
        
        if high_quality_faces:
            hq_labels, hq_info = self._cluster_face_subset(face_infos, high_quality_faces, current_cluster_id)
            for i, face_idx in enumerate(high_quality_faces):
                cluster_labels[face_idx] = hq_labels[i]
            
            cluster_info.update(hq_info)
            current_cluster_id = max(hq_labels) + 1 if len(hq_labels) > 0 else 0
        
        # Step 2: Assign medium and low quality faces to existing clusters or create new ones
        remaining_faces = []
        for quality in [FaceQuality.FAIR, FaceQuality.POOR]:
            if quality in quality_groups:
                remaining_faces.extend(quality_groups[quality])
        
        for face_idx in remaining_faces:
            best_cluster = self._find_best_cluster_for_face(face_infos[face_idx], face_infos, cluster_labels)
            
            if best_cluster is not None:
                cluster_labels[face_idx] = best_cluster
            else:
                # Create new cluster
                cluster_labels[face_idx] = current_cluster_id
                cluster_info[current_cluster_id] = {
                    'quality_distribution': Counter([face_infos[face_idx].quality_level]),
                    'avg_confidence': face_infos[face_idx].confidence,
                    'representative_idx': face_idx
                }
                current_cluster_id += 1
        
        return cluster_labels, cluster_info

    def _cluster_face_subset(self, all_faces: List[FaceInfo], face_indices: List[int], start_cluster_id: int) -> Tuple[List[int], Dict]:
        """
        Cluster a subset of faces using adaptive thresholds.
        """
        if not face_indices:
            return [], {}
        
        subset_faces = [all_faces[i] for i in face_indices]
        encodings = np.array([face.encoding for face in subset_faces])
        
        # Calculate pairwise distances with adaptive thresholds
        distance_matrix = self._calculate_adaptive_distance_matrix(subset_faces)
        
        # Use AgglomerativeClustering with precomputed distances
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,  # This will be overridden by our adaptive approach
            linkage='average',
            metric='precomputed'
        )
        
        # Custom clustering with adaptive thresholds
        labels = self._adaptive_agglomerative_clustering(subset_faces, distance_matrix)
        
        # Adjust labels to start from start_cluster_id
        adjusted_labels = [label + start_cluster_id if label >= 0 else -1 for label in labels]
        
        # Generate cluster info
        cluster_info = {}
        for i, label in enumerate(adjusted_labels):
            if label >= 0:
                if label not in cluster_info:
                    cluster_info[label] = {
                        'quality_distribution': Counter(),
                        'avg_confidence': 0.0,
                        'representative_idx': face_indices[i],
                        'members': []
                    }
                
                cluster_info[label]['quality_distribution'][subset_faces[i].quality_level] += 1
                cluster_info[label]['members'].append(face_indices[i])
        
        # Calculate average confidence for each cluster
        for cluster_id, info in cluster_info.items():
            confidences = [all_faces[idx].confidence for idx in info['members']]
            info['avg_confidence'] = np.mean(confidences)
            
            # Choose representative face (highest quality and confidence)
            best_idx = max(info['members'], 
                          key=lambda idx: (all_faces[idx].quality_level.value, all_faces[idx].confidence))
            info['representative_idx'] = best_idx
        
        return adjusted_labels, cluster_info

    def _calculate_adaptive_distance_matrix(self, faces: List[FaceInfo]) -> np.ndarray:
        """
        Calculate distance matrix with adaptive thresholds between faces.
        """
        n_faces = len(faces)
        distance_matrix = np.zeros((n_faces, n_faces))
        
        for i in range(n_faces):
            for j in range(i + 1, n_faces):
                # Calculate face_recognition distance
                face_distance = face_recognition.face_distance([faces[i].encoding], faces[j].encoding)[0]
                
                # Get adaptive threshold for this pair
                adaptive_threshold = self.adaptive_threshold.get_threshold(faces[i], faces[j])
                
                # Normalize distance by threshold (values > 1.0 are dissimilar)
                normalized_distance = face_distance / adaptive_threshold
                
                distance_matrix[i, j] = normalized_distance
                distance_matrix[j, i] = normalized_distance
        
        return distance_matrix

    def _adaptive_agglomerative_clustering(self, faces: List[FaceInfo], distance_matrix: np.ndarray) -> List[int]:
        """
        Custom agglomerative clustering with adaptive stopping criteria.
        """
        n_faces = len(faces)
        if n_faces <= 1:
            return [0] * n_faces
        
        # Initialize each face as its own cluster
        clusters = {i: [i] for i in range(n_faces)}
        cluster_labels = list(range(n_faces))
        
        while len(clusters) > 1:
            # Find the closest pair of clusters
            min_distance = float('inf')
            merge_pair = None
            
            cluster_ids = list(clusters.keys())
            for i, cluster_id1 in enumerate(cluster_ids):
                for cluster_id2 in cluster_ids[i + 1:]:
                    # Calculate average distance between clusters
                    distances = []
                    for face1_idx in clusters[cluster_id1]:
                        for face2_idx in clusters[cluster_id2]:
                            distances.append(distance_matrix[face1_idx, face2_idx])
                    
                    avg_distance = np.mean(distances)
                    
                    if avg_distance < min_distance:
                        min_distance = avg_distance
                        merge_pair = (cluster_id1, cluster_id2)
            
            # Stop if the minimum distance is too large (adaptive stopping)
            if min_distance > 1.0:  # Normalized distance > 1.0 means dissimilar
                break
            
            # Merge the closest clusters
            if merge_pair:
                cluster_id1, cluster_id2 = merge_pair
                
                # Merge cluster_id2 into cluster_id1
                clusters[cluster_id1].extend(clusters[cluster_id2])
                
                # Update labels
                for face_idx in clusters[cluster_id2]:
                    cluster_labels[face_idx] = cluster_id1
                
                # Remove the merged cluster
                del clusters[cluster_id2]
        
        # Reassign cluster IDs to be sequential starting from 0
        unique_clusters = sorted(set(cluster_labels))
        cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        
        return [cluster_mapping[label] for label in cluster_labels]

    def _find_best_cluster_for_face(self, target_face: FaceInfo, all_faces: List[FaceInfo], cluster_labels: np.ndarray) -> Optional[int]:
        """
        Find the best existing cluster for a target face.
        """
        existing_clusters = set(cluster_labels[cluster_labels >= 0])
        if not existing_clusters:
            return None
        
        best_cluster = None
        best_score = float('inf')
        
        for cluster_id in existing_clusters:
            cluster_face_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Calculate average distance to faces in this cluster
            distances = []
            for face_idx in cluster_face_indices:
                distance = face_recognition.face_distance([target_face.encoding], all_faces[face_idx].encoding)[0]
                adaptive_threshold = self.adaptive_threshold.get_threshold(target_face, all_faces[face_idx])
                normalized_distance = distance / adaptive_threshold
                distances.append(normalized_distance)
            
            avg_distance = np.mean(distances)
            
            # Consider cluster quality and confidence
            cluster_faces = [all_faces[idx] for idx in cluster_face_indices]
            avg_cluster_confidence = np.mean([face.confidence for face in cluster_faces])
            
            # Weighted score considering distance and confidence
            score = avg_distance * (2.0 - avg_cluster_confidence)  # Lower is better
            
            if score < best_score and avg_distance < 1.0:  # Only consider if similar enough
                best_score = score
                best_cluster = cluster_id
        
        return best_cluster

    def _adaptive_dbscan_clustering(self, face_infos: List[FaceInfo]) -> Tuple[np.ndarray, Dict]:
        """
        DBSCAN clustering with adaptive parameters based on face quality.
        """
        if not face_infos:
            return np.array([]), {}
        
        encodings = np.array([face.encoding for face in face_infos])
        
        # Calculate adaptive eps based on face quality distribution
        quality_scores = [face.quality_score for face in face_infos]
        confidence_scores = [face.confidence for face in face_infos]
        
        avg_quality = np.mean(quality_scores)
        avg_confidence = np.mean(confidence_scores)
        
        # Adaptive eps: higher quality/confidence -> stricter clustering
        base_eps = 0.5
        quality_adjustment = (1.0 - avg_quality) * 0.2
        confidence_adjustment = (1.0 - avg_confidence) * 0.1
        
        adaptive_eps = base_eps + quality_adjustment + confidence_adjustment
        adaptive_eps = np.clip(adaptive_eps, 0.3, 0.8)  # Reasonable bounds
        
        # Use DBSCAN with adaptive parameters
        clusterer = DBSCAN(eps=adaptive_eps, min_samples=self.min_cluster_size, metric='euclidean')
        cluster_labels = clusterer.fit_predict(encodings)
        
        # Generate cluster info
        cluster_info = {}
        for i, label in enumerate(cluster_labels):
            if label >= 0:  # Not noise
                if label not in cluster_info:
                    cluster_info[label] = {
                        'quality_distribution': Counter(),
                        'avg_confidence': 0.0,
                        'representative_idx': i,
                        'members': []
                    }
                
                cluster_info[label]['quality_distribution'][face_infos[i].quality_level] += 1
                cluster_info[label]['members'].append(i)
        
        # Calculate statistics for each cluster
        for cluster_id, info in cluster_info.items():
            confidences = [face_infos[idx].confidence for idx in info['members']]
            info['avg_confidence'] = np.mean(confidences)
            
            # Choose representative face
            best_idx = max(info['members'], 
                          key=lambda idx: (face_infos[idx].quality_level.value, face_infos[idx].confidence))
            info['representative_idx'] = best_idx
        
        return cluster_labels, cluster_info

    def _post_process_clusters(self, face_infos: List[FaceInfo], cluster_labels: np.ndarray, cluster_info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Post-process clusters to handle edge cases and improve accuracy.
        """
        # 1. Merge very similar clusters
        cluster_labels, cluster_info = self._merge_similar_clusters(face_infos, cluster_labels, cluster_info)
        
        # 2. Split clusters that are too diverse
        cluster_labels, cluster_info = self._split_diverse_clusters(face_infos, cluster_labels, cluster_info)
        
        # 3. Handle singleton clusters (clusters with only one face)
        cluster_labels, cluster_info = self._handle_singleton_clusters(face_infos, cluster_labels, cluster_info)
        
        return cluster_labels, cluster_info

    def _merge_similar_clusters(self, face_infos: List[FaceInfo], cluster_labels: np.ndarray, cluster_info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Merge clusters that are very similar based on representative faces.
        """
        cluster_ids = list(cluster_info.keys())
        if len(cluster_ids) < 2:
            return cluster_labels, cluster_info
        
        merge_candidates = []
        
        # Find merge candidates
        for i, cluster_id1 in enumerate(cluster_ids):
            for cluster_id2 in cluster_ids[i + 1:]:
                rep1_idx = cluster_info[cluster_id1]['representative_idx']
                rep2_idx = cluster_info[cluster_id2]['representative_idx']
                
                rep1_face = face_infos[rep1_idx]
                rep2_face = face_infos[rep2_idx]
                
                distance = face_recognition.face_distance([rep1_face.encoding], rep2_face.encoding)[0]
                threshold = self.adaptive_threshold.get_threshold(rep1_face, rep2_face)
                
                # More aggressive merging threshold for representatives
                if distance < threshold * 0.8:
                    merge_candidates.append((cluster_id1, cluster_id2, distance))
        
        # Sort by distance and merge
        merge_candidates.sort(key=lambda x: x[2])
        
        for cluster_id1, cluster_id2, _ in merge_candidates:
            if cluster_id1 in cluster_info and cluster_id2 in cluster_info:
                # Merge cluster_id2 into cluster_id1
                cluster_info[cluster_id1]['members'].extend(cluster_info[cluster_id2]['members'])
                
                # Update quality distribution
                for quality, count in cluster_info[cluster_id2]['quality_distribution'].items():
                    cluster_info[cluster_id1]['quality_distribution'][quality] += count
                
                # Update cluster labels
                for member_idx in cluster_info[cluster_id2]['members']:
                    cluster_labels[member_idx] = cluster_id1
                
                # Remove merged cluster
                del cluster_info[cluster_id2]
                
                # Recalculate statistics for merged cluster
                confidences = [face_infos[idx].confidence for idx in cluster_info[cluster_id1]['members']]
                cluster_info[cluster_id1]['avg_confidence'] = np.mean(confidences)
                
                best_idx = max(cluster_info[cluster_id1]['members'], 
                              key=lambda idx: (face_infos[idx].quality_level.value, face_infos[idx].confidence))
                cluster_info[cluster_id1]['representative_idx'] = best_idx
        
        return cluster_labels, cluster_info

    def _split_diverse_clusters(self, face_infos: List[FaceInfo], cluster_labels: np.ndarray, cluster_info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Split clusters that contain faces that are too different from each other.
        """
        clusters_to_split = []
        
        for cluster_id, info in cluster_info.items():
            if len(info['members']) < 3:  # Don't split small clusters
                continue
            
            # Calculate intra-cluster distances
            member_indices = info['members']
            distances = []
            
            for i, idx1 in enumerate(member_indices):
                for idx2 in member_indices[i + 1:]:
                    distance = face_recognition.face_distance([face_infos[idx1].encoding], face_infos[idx2].encoding)[0]
                    distances.append(distance)
            
            # If max distance is much larger than median, consider splitting
            if distances:
                max_distance = max(distances)
                median_distance = np.median(distances)
                
                if max_distance > median_distance * 2.0 and max_distance > 0.6:
                    clusters_to_split.append(cluster_id)
        
        # Split identified clusters
        next_cluster_id = max(cluster_info.keys()) + 1 if cluster_info else 0
        
        for cluster_id in clusters_to_split:
            member_indices = cluster_info[cluster_id]['members']
            member_faces = [face_infos[idx] for idx in member_indices]
            
            # Re-cluster this subset with stricter parameters
            sub_labels, sub_info = self._cluster_face_subset(face_infos, member_indices, next_cluster_id)
            
            # Update main cluster labels
            for i, member_idx in enumerate(member_indices):
                cluster_labels[member_idx] = sub_labels[i]
            
            # Remove old cluster info and add new ones
            del cluster_info[cluster_id]
            cluster_info.update(sub_info)
            
            if sub_info:
                next_cluster_id = max(sub_info.keys()) + 1
        
        return cluster_labels, cluster_info

    def _handle_singleton_clusters(self, face_infos: List[FaceInfo], cluster_labels: np.ndarray, cluster_info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Handle clusters with only one member - try to merge with nearby clusters or keep separate.
        """
        singleton_clusters = [cluster_id for cluster_id, info in cluster_info.items() if len(info['members']) == 1]
        
        for cluster_id in singleton_clusters:
            member_idx = cluster_info[cluster_id]['members'][0]
            target_face = face_infos[member_idx]
            
            # Only merge singletons if they're low quality
            if target_face.quality_level in [FaceQuality.POOR, FaceQuality.FAIR]:
                best_cluster = self._find_best_cluster_for_face(target_face, face_infos, cluster_labels)
                
                if best_cluster is not None and best_cluster != cluster_id:
                    # Merge into best cluster
                    cluster_labels[member_idx] = best_cluster
                    cluster_info[best_cluster]['members'].append(member_idx)
                    cluster_info[best_cluster]['quality_distribution'][target_face.quality_level] += 1
                    
                    # Remove singleton cluster
                    del cluster_info[cluster_id]
                    
                    # Recalculate statistics for the receiving cluster
                    confidences = [face_infos[idx].confidence for idx in cluster_info[best_cluster]['members']]
                    cluster_info[best_cluster]['avg_confidence'] = np.mean(confidences)
        
        return cluster_labels, cluster_info

    def get_merge_suggestions(self, face_infos: List[FaceInfo], cluster_labels: np.ndarray, cluster_info: Dict, threshold: float = 0.4) -> List[Tuple[int, int, float]]:
        """
        Suggest clusters that might be the same person and could be merged.
        """
        suggestions = []
        cluster_ids = list(cluster_info.keys())
        
        for i, cluster_id1 in enumerate(cluster_ids):
            for cluster_id2 in cluster_ids[i + 1:]:
                rep1_idx = cluster_info[cluster_id1]['representative_idx']
                rep2_idx = cluster_info[cluster_id2]['representative_idx']
                
                rep1_face = face_infos[rep1_idx]
                rep2_face = face_infos[rep2_idx]
                
                distance = face_recognition.face_distance([rep1_face.encoding], rep2_face.encoding)[0]
                adaptive_threshold = self.adaptive_threshold.get_threshold(rep1_face, rep2_face)
                
                # Suggest if close but not automatically merged
                if distance < adaptive_threshold * 1.2 and distance > adaptive_threshold * 0.8:
                    confidence_score = 1.0 - (distance / adaptive_threshold)
                    suggestions.append((cluster_id1, cluster_id2, confidence_score))
        
        # Sort by confidence score (higher is more confident)
        suggestions.sort(key=lambda x: x[2], reverse=True)
        
        return suggestions[:10]  # Return top 10 suggestions