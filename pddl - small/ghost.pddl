(define (domain ghost)
  (:requirements :typing :conditional-effects)
  (:types pos time)				;; Coordinate of a position as one variable
  (:predicates  (At ?p - pos) 		;; Position of Ghost
                (PacmanAt ?p - pos)	;; Position of Pacman
                (Adjacent ?p1 ?p2 - pos)	;; Whether two positions are connected
                (ScaredTime ?t - time)
                (TimeCounter ?t1 ?t2 -time)
                (notScaredTime ?notScared -time)
  )

  (:action move
        :parameters (?curPos ?nextPos - pos ?curTime -time)
        :precondition (and (At ?curPos)
                           (Adjacent ?curPos ?nextPos)
                           (ScaredTime ?curTime)
                           (notScaredTime ?curTime)
                       )
        :effect   (and (At ?nextPos)
                       (not  (At ?curPos))
                       (not  (PacmanAt ?nextPos) )
                   )
   )
   
   (:action stopAndWait
        :parameters (?curTime ?nextTime -time)
        :precondition (and (TimeCounter ?curTime ?nextTime)
                            ( ScaredTime ?curTime)
                            ( not (notScaredTime ?curTime)
                            
                       ))
        :effect   (and (ScaredTime ?nextTime)   
                       (not  (ScaredTime ?curTime) )
                       )
                   )
   )
